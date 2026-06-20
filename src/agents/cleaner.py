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
            "IMPORTANT: A pandas DataFrame named `df` ALREADY EXISTS in the "
            "execution environment, pre-loaded with the real uploaded data. It is "
            "injected automatically before your code runs. Do NOT create your own "
            "sample/example/mock data. Do NOT write `df = pd.DataFrame(...)` or "
            "`data = [...]` with hardcoded values. Only write transformation code "
            "that operates on the existing `df` variable as-is.\n\n"
            "Analyze the following dataset schema and write Python/pandas code to clean it.\n\n"
            f"{schema_text}\n\n"
            "Your cleaning code MUST:\n"
            "1. Standardize column names (lowercase, strip spaces, replace spaces with underscores)\n"
            "2. Parse date columns to datetime SAFELY using this exact pattern, which "
            "handles mixed formats (e.g. some rows as 'YYYY-MM-DD', others as "
            "'MM/DD/YYYY') without losing data to NaT:\n"
            "   ```\n"
            "   def parse_mixed_date(val):\n"
            "       if pd.isna(val):\n"
            "           return pd.NaT\n"
            "       for fmt in ('%Y-%m-%d', '%m/%d/%Y', '%d/%m/%Y', '%Y/%m/%d'):\n"
            "           try:\n"
            "               return pd.to_datetime(val, format=fmt)\n"
            "           except (ValueError, TypeError):\n"
            "               continue\n"
            "       return pd.to_datetime(val, errors='coerce')\n"
            "   df['date'] = df['date'].apply(parse_mixed_date)\n"
            "   ```\n"
            "   Do NOT use `format='mixed'` or `infer_datetime_format=True` alone — "
            "they are unreliable and silently drop valid dates to NaT. Use the "
            "explicit per-row format-trying approach above instead.\n"
            "   After parsing, check `df['date'].isnull().sum()` and print a warning "
            "if any dates remain unparsed, but DO NOT let this happen for dates that "
            "are validly formatted — verify your logic actually parses both "
            "'YYYY-MM-DD' and 'MM/DD/YYYY' style strings correctly.\n"
            "3. Fill missing numeric values with median, categorical with mode\n"
            "4. Convert data types appropriately\n"
            "5. Remove duplicate rows\n"
            "6. Assign the cleaned DataFrame to variable `df` (reusing the existing `df`, "
            "not replacing it with new data)\n"
            "7. Print a summary of changes made\n\n"
            "Return ONLY executable Python code inside a ```python code block. "
            "Do NOT include any import statements for pandas, numpy, etc. — they "
            "are already available as `pd` and `np`."
        ),
        expected_output=(
            "Executable Python code in a ```python block that cleans the existing "
            "`df` (never replaces it with sample data) and assigns the result back "
            "to `df`, with ALL valid dates correctly parsed (none lost to NaT due "
            "to mixed formats)."
        ),
        agent=agent,
    )