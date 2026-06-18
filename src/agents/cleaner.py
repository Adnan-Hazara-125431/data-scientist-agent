"""Data Cleaner agent — fixes dates, fills missing values, standardizes columns."""

from crewai import Agent, Task

from .llm import get_llm


def create_cleaner_agent() -> Agent:
    """Create the Data Cleaner CrewAI agent."""
    return Agent(
        role="Data Cleaner",
        goal=(
            "Clean messy datasets by fixing date formats, filling missing values, "
            "standardizing column names, and correcting data types."
        ),
        backstory=(
            "You are an expert data engineer with years of experience cleaning "
            "real-world messy datasets. You write precise, executable pandas code "
            "that transforms raw data into analysis-ready DataFrames. You always "
            "assign the final cleaned DataFrame to a variable named `df`."
        ),
        llm=get_llm(temperature=0.1),
        verbose=True,
        allow_delegation=False,
    )


def create_cleaning_task(agent: Agent, schema_text: str) -> Task:
    """Create a task for the cleaner agent."""
    return Task(
        description=(
            "Analyze the following dataset schema and write Python/pandas code to clean it.\n\n"
            f"{schema_text}\n\n"
            "Your cleaning code MUST:\n"
            "1. Standardize column names (lowercase, strip spaces, replace spaces with underscores)\n"
            "2. Parse date columns to datetime where applicable\n"
            "3. Fill missing numeric values with median, categorical with mode\n"
            "4. Convert data types appropriately\n"
            "5. Remove duplicate rows\n"
            "6. Assign the cleaned DataFrame to variable `df`\n"
            "7. Print a summary of changes made\n\n"
            "Return ONLY executable Python code inside a ```python code block."
        ),
        expected_output=(
            "Executable Python code in a ```python block that cleans the data "
            "and assigns the result to `df`."
        ),
        agent=agent,
    )
