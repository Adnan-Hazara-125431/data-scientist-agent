"""Code Healer agent — executes code, catches errors, retries up to 5 times."""

from __future__ import annotations

from crewai import Agent, Crew, Process, Task

from ..tools.code_executor import execute_code, extract_python_code

from .llm import get_llm

MAX_RETRIES = 5


def create_code_healer_agent() -> Agent:
    """Create the Code Healer CrewAI agent."""
    return Agent(
        role="Code Healer / Debugger",
        goal=(
            "Execute Python data analysis code, diagnose errors, and produce fixed "
            "code that runs successfully."
        ),
        backstory=(
            "You are an expert Python debugger specializing in pandas data analysis. "
            "When code fails, you carefully read the traceback, identify the root cause, "
            "and produce corrected code. You never give up until the code works or "
            "you have exhausted all reasonable fixes."
        ),
        llm=get_llm(temperature=0.1),
        verbose=True,
        allow_delegation=False,
    )


def _build_heal_prompt(code: str, error: str, context: str) -> str:
    return (
        f"The following Python/pandas code failed with an error.\n\n"
        f"Context: {context}\n\n"
        f"```python\n{code}\n```\n\n"
        f"Error traceback:\n{error}\n\n"
        "Fix the code and return ONLY the corrected executable Python code "
        "inside a ```python code block. Keep the same intent. "
        "Ensure `df` is used as input and `result_df` holds the output."
    )


def heal_code(
    agent: Agent,
    code: str,
    df,
    context: str = "analysis",
    max_retries: int = MAX_RETRIES,
) -> dict:
    """
    Execute code with self-healing retry loop.

    Returns dict with: success, code, result, attempts, stdout, error.
    """
    current_code = extract_python_code(code)
    last_error = None
    last_stdout = ""

    for attempt in range(1, max_retries + 1):
        result = execute_code(current_code, df=df)
        last_stdout = result["stdout"]

        if result["success"]:
            return {
                "success": True,
                "code": current_code,
                "result": result,
                "attempts": attempt,
                "stdout": last_stdout,
                "error": None,
            }

        last_error = result["error"]
        if attempt >= max_retries:
            break

        heal_prompt = _build_heal_prompt(current_code, last_error, context)
        heal_task = Task(
            description=heal_prompt,
            expected_output="Corrected executable Python code in a ```python block.",
            agent=agent,
        )
        heal_crew = Crew(
            agents=[agent],
            tasks=[heal_task],
            process=Process.sequential,
            verbose=True,
        )
        healed_response = heal_crew.kickoff()
        healed_text = str(healed_response)
        current_code = extract_python_code(healed_text)

    return {
        "success": False,
        "code": current_code,
        "result": None,
        "attempts": max_retries,
        "stdout": last_stdout,
        "error": last_error,
    }
