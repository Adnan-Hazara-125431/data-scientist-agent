"""
Streamlit UI for the Data Scientist Agent.

Run from project root:
    streamlit run src/app.py
"""

from __future__ import annotations

import re
import sys
import tempfile
from pathlib import Path
from typing import Callable

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agents.orchestrator import DataScientistOrchestrator

st.set_page_config(
    page_title="Data Scientist Agent",
    page_icon="📊",
    layout="wide",
)

# Regex to strip ANSI escape codes (e.g. from CrewAI's colored terminal output)
ANSI_ESCAPE_PATTERN = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]")

# Matches CrewAI's entire "Tracing Status" notice box, including the
# top/bottom border lines that contain the "Tracing Status" text itself
# (e.g. "╭─────── Tracing Status ───────╮"), the info lines inside the
# box, and the closing border line. This is more reliable than only
# matching pure box-drawing lines, since the border lines mix box
# characters with the "Tracing Status" label text.
TRACING_BOX_PATTERN = re.compile(
    r"[╭┌].*?Tracing Status.*?[╰└][─\s]*[╯┘]?",
    re.DOTALL,
)

# Fallback: matches any remaining line made up entirely of box-drawing
# characters (covers stray border fragments not caught above).
BOX_LINE_PATTERN = re.compile(r"^[\s│╭╮╰╯─┌┐└┘|]+$", re.MULTILINE)

# Matches the "Tracing is disabled" notice text itself, in case it
# appears outside of a box (plain text fallback).
TRACING_TEXT_PATTERN = re.compile(
    r"Info: Tracing is disabled\..*?crewai traces enable",
    re.DOTALL,
)


def clean_text(text: str) -> str:
    """Remove ANSI codes, CrewAI's tracing notice box, and escape $ for markdown."""
    if not text:
        return text
    text = ANSI_ESCAPE_PATTERN.sub("", text)
    text = TRACING_BOX_PATTERN.sub("", text)
    text = TRACING_TEXT_PATTERN.sub("", text)
    text = BOX_LINE_PATTERN.sub("", text)
    text = text.replace("$", "\\$")
    # Collapse multiple blank lines left behind after removal
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


st.title("📊 Data Scientist Agent")
st.markdown(
    "Upload a messy CSV or Excel file, ask a question, and let the AI agents "
    "**clean**, **analyze**, **self-heal**, and **visualize** your data."
)

# Sidebar
with st.sidebar:
    st.header("How it works")
    st.markdown(
        """
        1. **Orchestrator** — manages the workflow
        2. **Data Cleaner** — fixes dates, fills missing values
        3. **Code Writer** — writes pandas analysis code
        4. **Code Healer** — fixes errors (up to 5 retries)
        5. **Visualizer** — creates charts & summary
        """
    )
    st.divider()
    st.markdown("**Supported formats:** CSV, Excel (.xlsx, .xls)")
    st.markdown("**LLM:** Google Gemini via CrewAI")

# Main inputs
col1, col2 = st.columns([1, 1])

with col1:
    uploaded_file = st.file_uploader(
        "Upload your data file",
        type=["csv", "xlsx", "xls"],
        help="Upload a CSV or Excel file to analyze",
    )

with col2:
    question = st.text_area(
        "Your analysis question",
        placeholder="e.g., What is the average revenue by region?",
        height=100,
    )

analyze_btn = st.button("🔍 Analyze", type="primary", use_container_width=True)

if analyze_btn:
    if not uploaded_file:
        st.error("Please upload a data file first.")
    elif not question.strip():
        st.error("Please enter an analysis question.")
    else:
        suffix = Path(uploaded_file.name).suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_path = tmp.name

        try:
            orchestrator = DataScientistOrchestrator()
            status_box = st.empty()
            progress_lines: list[str] = []

            def on_progress(msg: str) -> None:
                progress_lines.append(msg)
                status_box.markdown("\n\n".join(f"- {line}" for line in progress_lines))

            with st.spinner("Running multi-agent pipeline..."):
                result = orchestrator.run(
                    file_path=tmp_path,
                    question=question.strip(),
                    on_progress=on_progress,
                )

            if result.success:
                st.success("Analysis completed successfully!")
            else:
                st.warning("Analysis completed with some errors.")

            # Cleaned data preview
            if result.cleaned_df is not None:
                st.subheader("🧹 Cleaned Data Preview")
                st.dataframe(result.cleaned_df.head(20), use_container_width=True)
                st.caption(
                    f"Shape: {result.cleaned_df.shape[0]} rows × "
                    f"{result.cleaned_df.shape[1]} columns"
                )

            # Analysis output
            if result.analysis_output:
                st.subheader("📋 Analysis Output")
                st.code(clean_text(result.analysis_output), language="text")

            # Charts
            if result.chart_paths:
                st.subheader("📈 Visualizations")
                chart_cols = st.columns(min(len(result.chart_paths), 3))
                for i, chart_path in enumerate(result.chart_paths):
                    with chart_cols[i % len(chart_cols)]:
                        st.image(chart_path, use_container_width=True)

            # Summary
            if result.summary:
                st.subheader("📝 Summary")
                st.markdown(clean_text(result.summary))

            # Errors
            if result.errors:
                with st.expander("⚠️ Errors & Warnings"):
                    for err in result.errors:
                        st.warning(clean_text(err))

            # Code expanders
            with st.expander("🔍 Generated Code"):
                if result.cleaning_code:
                    st.markdown("**Cleaning Code**")
                    st.code(result.cleaning_code, language="python")
                if result.analysis_code:
                    st.markdown("**Analysis Code**")
                    st.code(result.analysis_code, language="python")
                if result.visualization_code:
                    st.markdown("**Visualization Code**")
                    st.code(result.visualization_code, language="python")

        except ValueError as exc:
            st.error(str(exc))
            st.info("Make sure your GEMINI_API_KEY is set in the .env file.")
        except Exception as exc:
            st.error(f"An unexpected error occurred: {exc}")
        finally:
            Path(tmp_path).unlink(missing_ok=True)

else:
    st.info("Upload a file and enter a question, then click **Analyze** to begin.")