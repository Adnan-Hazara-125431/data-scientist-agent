"""Generate and save matplotlib charts."""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

matplotlib.use("Agg")

DEFAULT_OUTPUT_DIR = Path("output/charts")


def _ensure_output_dir(output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def save_figure(
    fig: plt.Figure,
    filename: str | None = None,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> str:
    """Save a matplotlib figure to disk and return the file path."""
    out_dir = _ensure_output_dir(Path(output_dir))
    if filename is None:
        filename = f"chart_{uuid.uuid4().hex[:8]}.png"
    filepath = out_dir / filename
    fig.savefig(filepath, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return str(filepath)


def generate_chart(
    df: pd.DataFrame,
    chart_type: str,
    x_col: str | None = None,
    y_col: str | None = None,
    title: str = "Data Visualization",
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    """
    Generate a chart from a DataFrame.

    Supported chart_type values: bar, line, scatter, histogram, box, heatmap.
    """
    _ensure_output_dir(Path(output_dir))
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6))

    chart_type = chart_type.lower().strip()

    try:
        if chart_type == "bar" and x_col and y_col:
            sns.barplot(data=df, x=x_col, y=y_col, ax=ax)
        elif chart_type == "line" and x_col and y_col:
            sns.lineplot(data=df, x=x_col, y=y_col, ax=ax)
        elif chart_type == "scatter" and x_col and y_col:
            sns.scatterplot(data=df, x=x_col, y=y_col, ax=ax)
        elif chart_type == "histogram" and x_col:
            sns.histplot(data=df, x=x_col, ax=ax, kde=True)
        elif chart_type == "box" and x_col:
            cols = [x_col] if y_col is None else [x_col, y_col]
            plot_df = df[cols].dropna()
            if y_col:
                sns.boxplot(data=plot_df, x=x_col, y=y_col, ax=ax)
            else:
                sns.boxplot(data=plot_df, y=x_col, ax=ax)
        elif chart_type == "heatmap":
            numeric_df = df.select_dtypes(include="number")
            if numeric_df.empty:
                raise ValueError("No numeric columns available for heatmap.")
            sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", ax=ax)
        else:
            numeric_cols = df.select_dtypes(include="number").columns.tolist()
            if numeric_cols:
                df[numeric_cols[0]].plot(kind="hist", ax=ax, bins=20)
            else:
                df.iloc[:, 0].value_counts().head(10).plot(kind="bar", ax=ax)

        ax.set_title(title)
        plt.tight_layout()
        filepath = save_figure(fig, output_dir=output_dir)
        return {"success": True, "filepath": filepath, "chart_type": chart_type, "error": None}
    except Exception as exc:
        plt.close(fig)
        return {"success": False, "filepath": None, "chart_type": chart_type, "error": str(exc)}


def save_all_figures(
    figures: list[plt.Figure],
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    prefix: str = "chart",
) -> list[str]:
    """Save multiple matplotlib figures and return file paths."""
    paths = []
    for i, fig in enumerate(figures):
        path = save_figure(fig, filename=f"{prefix}_{i + 1}.png", output_dir=output_dir)
        paths.append(path)
    return paths
