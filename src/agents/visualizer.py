"""Visualizer agent — creates charts and writes human-readable summaries."""

from crewai import Agent, Task

from .llm import get_llm


def create_visualizer_agent() -> Agent:
    """Create the Visualizer CrewAI agent."""
    return Agent(
        role="Data Visualizer & Storyteller",
        goal=(
            "Create insightful matplotlib/seaborn visualizations and write clear, "
            "human-readable summaries of data analysis results."
        ),
        backstory=(
            "You are a data visualization expert and communicator. You create "
            "publication-quality charts using matplotlib and seaborn, and you "
            "explain findings in plain language that non-technical stakeholders "
            "can understand. You write code that creates charts and leaves them "
            "open for the calling system to save and display."
        ),
        llm=get_llm(temperature=0.3),
        verbose=True,
        allow_delegation=False,
    )


def create_visualization_task(
    agent: Agent,
    question: str,
    analysis_output: str,
    columns: list[str],
    previous_error: str = "",
) -> Task:
    """Create a task for the visualizer agent."""

    error_section = ""
    if previous_error:
        error_section = (
            f"PREVIOUS ATTEMPT FAILED WITH THIS ERROR:\n{previous_error}\n\n"
            "Fix the error above. Double-check your function call syntax "
            "carefully — watch for things like duplicate or malformed keyword "
            "arguments (e.g. alpha=0=0.7 is WRONG, alpha=0.7 is correct).\n\n"
        )

    return Task(
        description=(
            "IMPORTANT: A pandas DataFrame named `df` and/or `result_df` ALREADY "
            "EXISTS in the execution environment with the real data. Do NOT create "
            "your own sample data.\n\n"
            "CRITICAL OUTPUT FORMAT RULE: Your ENTIRE response must be ONLY a single "
            "```python code block. Do NOT write any summary, explanation, or "
            "narrative text outside the code block — that comes in a separate step. "
            "If you have findings to highlight, print them using print() statements "
            "INSIDE the code block, not as plain text in your response.\n\n"
            "CRITICAL CHART RULE: Do NOT call plt.savefig() and do NOT call "
            "plt.close() anywhere in your code. Simply create each chart with "
            "plt.figure() and leave it open after drawing it — the system handles "
            "saving and displaying figures automatically after your code runs. "
            "Calling savefig or close yourself will break the chart display.\n\n"
            f"{error_section}"
            f"User question: {question}\n\n"
            f"Analysis output:\n{analysis_output}\n\n"
            f"Available columns: {', '.join(columns)}\n\n"
            "Write Python code that:\n"
            "1. Uses `df` or `result_df` (whichever is available in the namespace)\n"
            "2. Creates 1-3 meaningful matplotlib/seaborn charts\n"
            "3. Uses plt.figure() for each chart and leaves it open (no savefig, no close)\n"
            "4. Adds titles, labels, and legends\n"
            "5. Optionally prints key findings via print() inside the code\n\n"
            "Return ONLY the executable Python code inside a single ```python code "
            "block. Nothing before it, nothing after it."
        ),
        expected_output=(
            "A single ```python code block containing executable visualization "
            "code that leaves figures open (no savefig/close calls), with no text "
            "outside the code block."
        ),
        agent=agent,
    )


def create_summary_task(
    agent: Agent,
    question: str,
    analysis_output: str,
    chart_info: str,
) -> Task:
    """Create a task to generate a final narrative summary."""
    return Task(
        description=(
            f"User question: {question}\n\n"
            f"Analysis results:\n{analysis_output}\n\n"
            f"Charts created:\n{chart_info}\n\n"
            "Write a clear, human-readable summary (200-400 words) that:\n"
            "1. Directly answers the user's question\n"
            "2. Highlights the most important findings\n"
            "3. Mentions relevant statistics or trends\n"
            "4. Notes any limitations in the data\n"
            "5. Uses plain language — no code\n"
            "6. When writing dollar amounts, write them as 'USD 1234.50' instead "
            "of '$1234.50' to avoid markdown rendering issues\n"
        ),
        expected_output="A human-readable analysis summary in plain text.",
        agent=agent,
    )