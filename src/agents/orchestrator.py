"""Orchestrator — manages the full multi-agent data science pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import pandas as pd
from crewai import Crew, Process

from ..tools.chart_generator import save_all_figures
from ..tools.code_executor import execute_code, extract_python_code
from ..tools.data_loader import format_schema_for_prompt, get_schema_info, load_data

from .cleaner import create_cleaner_agent, create_cleaning_task
from .code_healer import create_code_healer_agent, heal_code
from .code_writer import create_code_writer_agent, create_analysis_task
from .visualizer import (
    create_summary_task,
    create_visualizer_agent,
    create_visualization_task,
)


@dataclass
class PipelineResult:
    """Container for all pipeline outputs."""

    success: bool
    question: str
    raw_df: pd.DataFrame
    cleaned_df: pd.DataFrame | None = None
    cleaning_code: str = ""
    analysis_code: str = ""
    visualization_code: str = ""
    analysis_output: str = ""
    summary: str = ""
    chart_paths: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    progress: list[str] = field(default_factory=list)


class DataScientistOrchestrator:
    """
    Orchestrates the 5-agent pipeline:
    Cleaner → Code Writer → Code Healer → Visualizer
    """

    def __init__(self, output_dir: str | Path = "output/charts"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.cleaner = create_cleaner_agent()
        self.code_writer = create_code_writer_agent()
        self.code_healer = create_code_healer_agent()
        self.visualizer = create_visualizer_agent()

    def _log(
        self,
        messages: list[str],
        msg: str,
        on_progress: Callable[[str], None] | None = None,
    ) -> None:
        messages.append(msg)
        print(msg)
        if on_progress:
            on_progress(msg)

    def run(
        self,
        file_path: str | Path | None = None,
        df: pd.DataFrame | None = None,
        question: str = "Summarize the key patterns in this dataset.",
        on_progress: Callable[[str], None] | None = None,
    ) -> PipelineResult:
        """Run the full analysis pipeline."""
        progress: list[str] = []
        errors: list[str] = []

        if df is None:
            if file_path is None:
                raise ValueError("Provide either file_path or df.")
            self._log(progress, f"[Orchestrator] Loading data from {file_path}...", on_progress)
            raw_df = load_data(file_path)
        else:
            self._log(progress, "[Orchestrator] Using provided DataFrame...", on_progress)
            raw_df = df.copy()

        result = PipelineResult(
            success=False,
            question=question,
            raw_df=raw_df,
            progress=progress,
            errors=errors,
        )

        schema = get_schema_info(raw_df)
        schema_text = format_schema_for_prompt(schema)

        # ── Step 1: Data Cleaner ──────────────────────────────────────────
        self._log(progress, "[Cleaner] Starting data cleaning...", on_progress)
        cleaning_task = create_cleaning_task(self.cleaner, schema_text)
        cleaning_crew = Crew(
            agents=[self.cleaner],
            tasks=[cleaning_task],
            process=Process.sequential,
            verbose=True,
        )
        cleaning_response = cleaning_crew.kickoff()
        cleaning_code = extract_python_code(str(cleaning_response))
        result.cleaning_code = cleaning_code

        clean_result = execute_code(cleaning_code, df=raw_df)
        if not clean_result["success"]:
            self._log(progress, "[Cleaner] Cleaning code failed, using raw data...", on_progress)
            errors.append(f"Cleaning failed: {clean_result['error']}")
            cleaned_df = raw_df.copy()
        else:
            cleaned_df = (
                clean_result["result_df"]
                if clean_result["result_df"] is not None
                else raw_df.copy()
            )
            self._log(progress, f"[Cleaner] Done. Shape: {cleaned_df.shape}", on_progress)

        result.cleaned_df = cleaned_df
        cleaned_schema = get_schema_info(cleaned_df)
        cleaned_schema_text = format_schema_for_prompt(cleaned_schema)

        # ── Step 2: Code Writer ───────────────────────────────────────────
        self._log(progress, "[Code Writer] Writing analysis code...", on_progress)
        analysis_task = create_analysis_task(
            self.code_writer,
            question,
            cleaned_schema_text,
            cleaned_schema["columns"],
        )
        writer_crew = Crew(
            agents=[self.code_writer],
            tasks=[analysis_task],
            process=Process.sequential,
            verbose=True,
        )
        analysis_response = writer_crew.kickoff()
        analysis_code = extract_python_code(str(analysis_response))
        result.analysis_code = analysis_code

        # ── Step 3: Code Healer (up to 5 retries) ─────────────────────────
        self._log(progress, "[Code Healer] Executing analysis code...", on_progress)
        heal_result = heal_code(
            self.code_healer,
            analysis_code,
            cleaned_df,
            context=f"Answer: {question}",
        )

        if not heal_result["success"]:
            self._log(
                progress,
                f"[Code Healer] Failed after {heal_result['attempts']} attempts.",
                on_progress,
            )
            errors.append(f"Analysis failed: {heal_result['error']}")
            result.analysis_output = heal_result.get("stdout", "")
            result.errors = errors
            result.progress = progress
            return result

        result.analysis_code = heal_result["code"]
        result.analysis_output = heal_result["stdout"]
        self._log(
            progress,
            f"[Code Healer] Success on attempt {heal_result['attempts']}.",
            on_progress,
        )

        analysis_df = (
            heal_result["result"]["result_df"]
            if heal_result["result"]["result_df"] is not None
            else cleaned_df
        )

        # ── Step 4: Visualizer (up to 3 retries) ─────────────────────────
        self._log(progress, "[Visualizer] Creating charts and summary...", on_progress)

        viz_exec = {"success": False, "error": "", "figures": [], "stdout": ""}
        viz_code = ""
        max_viz_attempts = 3
        previous_error = ""

        for attempt in range(1, max_viz_attempts + 1):
            viz_task = create_visualization_task(
                self.visualizer,
                question,
                result.analysis_output,
                list(analysis_df.columns),
                previous_error=previous_error,
            )
            viz_crew = Crew(
                agents=[self.visualizer],
                tasks=[viz_task],
                process=Process.sequential,
                verbose=True,
            )
            viz_response = viz_crew.kickoff()
            viz_code = extract_python_code(str(viz_response))

            viz_exec = execute_code(
                viz_code,
                df=cleaned_df,
                extra_locals={"result_df": analysis_df},
            )

            if viz_exec["success"]:
                self._log(
                    progress,
                    f"[Visualizer] Chart code succeeded on attempt {attempt}.",
                    on_progress,
                )
                break
            else:
                previous_error = viz_exec["error"]
                self._log(
                    progress,
                    f"[Visualizer] Chart attempt {attempt} failed, retrying...",
                    on_progress,
                )

        result.visualization_code = viz_code

        if viz_exec["success"] and viz_exec["figures"]:
            chart_paths = save_all_figures(
                viz_exec["figures"],
                output_dir=self.output_dir,
                prefix="analysis",
            )
            result.chart_paths = chart_paths
            self._log(progress, f"[Visualizer] Saved {len(chart_paths)} chart(s).", on_progress)
        elif not viz_exec["success"]:
            errors.append(f"Visualization code failed: {viz_exec['error']}")
            self._log(
                progress,
                "[Visualizer] Chart code failed after 3 attempts, generating summary only...",
                on_progress,
            )

        # Generate narrative summary.
        # NOTE: we intentionally do NOT prepend viz_exec["stdout"] here.
        # The visualization code's printed output (e.g. "Total Sales Revenue
        # by Region: ...") duplicates the analysis output already shown in
        # the "Analysis Output" section above, so mixing it into the
        # narrative summary just creates a confusing repeated intro.
        # The summary task below produces a clean, standalone narrative.
        chart_info = (
            ", ".join(result.chart_paths) if result.chart_paths else "No charts generated"
        )
        summary_task = create_summary_task(
            self.visualizer,
            question,
            result.analysis_output,
            chart_info,
        )
        summary_crew = Crew(
            agents=[self.visualizer],
            tasks=[summary_task],
            process=Process.sequential,
            verbose=True,
        )
        summary_response = summary_crew.kickoff()
        result.summary = str(summary_response).strip()

        result.success = True
        result.errors = errors
        result.progress = progress
        self._log(progress, "[Orchestrator] Pipeline complete!", on_progress)
        return result