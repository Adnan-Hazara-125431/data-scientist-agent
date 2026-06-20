"""Code Writer agent — writes pandas code to answer the user's question."""

from crewai import Agent, Task

from .llm import get_llm


def create_code_writer_agent() -> Agent:
    """Create the Code Writer CrewAI agent."""
    return Agent(
        role="Data Analyst / Code Writer",
        goal=(
            "Write clear, correct pandas code that answers the user's data analysis "
            "question using the cleaned dataset."
        ),
        backstory=(
            "You are a senior data scientist who writes production-quality pandas code. "
            "Your code is concise, well-commented, and stores analysis results in "
            "variables that can be inspected. You use `df` as the input DataFrame "
            "and create a `result_df` variable with the analysis output."
        ),
        llm=get_llm(temperature=0.2),
        verbose=True,
        allow_delegation=False,
    )


def create_analysis_task(
    agent: Agent,
    question: str,
    schema_text: str,
    columns: list[str],
) -> Task:
    """Create a task for the code writer agent."""
    return Task(
        description=(
            "IMPORTANT: A pandas DataFrame named `df` ALREADY EXISTS in the "
            "execution environment, pre-loaded with the real cleaned data. It is "
            "injected automatically before your code runs. Do NOT create your own "
            "sample/example/mock data. Do NOT write `df = pd.DataFrame(...)` or "
            "`data = [...]` with hardcoded values. Only write analysis code that "
            "operates on the existing `df` variable as-is.\n\n"
            f"User question: {question}\n\n"
            f"Available columns: {', '.join(columns)}\n\n"
            f"Dataset info:\n{schema_text}\n\n"
            "Write Python/pandas code that:\n"
            "1. Uses the existing `df` DataFrame (already cleaned, do not recreate it)\n"
            "2. Answers the user's question completely\n"
            "3. Stores the main result in a variable called `result_df`\n"
            "4. Prints key findings to stdout\n"
            "5. Uses only pandas, numpy — no file I/O or network calls\n\n"
            "Return ONLY executable Python code inside a ```python code block. "
            "Do NOT include any import statements for pandas, numpy, etc. — they "
            "are already available as `pd` and `np`."
        ),
        expected_output=(
            "Executable Python code in a ```python block that analyzes the "
            "existing `df` (never replaces it with sample data) and stores "
            "results in `result_df`."
        ),
        agent=agent,
    )