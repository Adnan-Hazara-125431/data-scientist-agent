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
            "can understand. You write code that saves charts using plt.savefig "
            "and prints a narrative summary."
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
) -> Task:
    """Create a task for the visualizer agent."""
    return Task(
        description=(
            f"User question: {question}\n\n"
            f"Analysis output:\n{analysis_output}\n\n"
            f"Available columns: {', '.join(columns)}\n\n"
            "Write Python code that:\n"
            "1. Uses `df` or `result_df` (whichever is available in the namespace)\n"
            "2. Creates 1-3 meaningful matplotlib/seaborn charts\n"
            "3. Uses plt.figure() for each chart\n"
            "4. Adds titles, labels, and legends\n"
            "5. Prints a human-readable SUMMARY section with:\n"
            "   - Key findings (3-5 bullet points)\n"
            "   - Answer to the user's question\n"
            "   - Any caveats or data quality notes\n\n"
            "Return ONLY executable Python code inside a ```python code block, "
            "followed by the text summary in a comment block at the end."
        ),
        expected_output=(
            "Executable Python visualization code in a ```python block, "
            "plus a plain-language summary of findings."
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
        ),
        expected_output="A human-readable analysis summary in plain text.",
        agent=agent,
    )
