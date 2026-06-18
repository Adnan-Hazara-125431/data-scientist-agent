"""Load CSV/Excel files and return dataframe with schema information."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def load_data(file_path: str | Path) -> pd.DataFrame:
    """Load a CSV or Excel file into a pandas DataFrame."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    raise ValueError(f"Unsupported file type: {suffix}. Use CSV or Excel.")


def get_schema_info(df: pd.DataFrame, sample_rows: int = 5) -> dict[str, Any]:
    """Return schema metadata and a small sample for agent context."""
    sample = df.head(sample_rows).to_dict(orient="records")
    dtypes = {col: str(dtype) for col, dtype in df.dtypes.items()}
    missing = df.isnull().sum().to_dict()
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    datetime_cols = df.select_dtypes(include=["datetime64"]).columns.tolist()

    return {
        "columns": list(df.columns),
        "dtypes": dtypes,
        "shape": list(df.shape),
        "missing_values": missing,
        "numeric_columns": numeric_cols,
        "categorical_columns": categorical_cols,
        "datetime_columns": datetime_cols,
        "sample_data": sample,
        "memory_usage_mb": round(df.memory_usage(deep=True).sum() / 1024 / 1024, 2),
    }


def format_schema_for_prompt(schema: dict[str, Any]) -> str:
    """Format schema info as readable text for LLM prompts."""
    lines = [
        f"Shape: {schema['shape'][0]} rows x {schema['shape'][1]} columns",
        f"Columns: {', '.join(schema['columns'])}",
        f"Data types: {schema['dtypes']}",
        f"Missing values: {schema['missing_values']}",
        f"Numeric columns: {schema['numeric_columns']}",
        f"Categorical columns: {schema['categorical_columns']}",
        f"Sample data (first rows): {schema['sample_data']}",
    ]
    return "\n".join(lines)
