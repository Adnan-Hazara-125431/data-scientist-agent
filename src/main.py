"""
Pipeline test with mock messy data.

Run from project root:
    python -m src.main
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agents.orchestrator import DataScientistOrchestrator


def create_mock_messy_data() -> pd.DataFrame:
    """Create a deliberately messy dataset for pipeline testing."""
    np.random.seed(42)
    n = 50

    data = {
        "Product Name": np.random.choice(
            ["Widget A", "Widget B", "Gadget X", None], n
        ),
        "Sale Date": [
            f"{d:02d}/{(m % 12) + 1:02d}/2024"
            for d, m in zip(
                np.random.randint(1, 29, n),
                np.random.randint(0, 12, n),
            )
        ],
        "Revenue ": np.random.uniform(100, 5000, n).round(2),
        "Units Sold": np.random.randint(1, 100, n).astype(float),
        "Region": np.random.choice(["North", "South", "East", "West", None], n),
    }

    df = pd.DataFrame(data)
    # Inject missing values
    df.loc[np.random.choice(n, 8, replace=False), "Revenue "] = np.nan
    df.loc[np.random.choice(n, 5, replace=False), "Units Sold"] = np.nan
    # Add duplicate rows
    df = pd.concat([df, df.iloc[:3]], ignore_index=True)
    return df


def main() -> None:
    print("=" * 60)
    print("Data Scientist Agent — Pipeline Test")
    print("=" * 60)

    print("\n[Setup] Creating mock messy dataset...")
    messy_df = create_mock_messy_data()
    print(f"  Raw shape: {messy_df.shape}")
    print(f"  Columns: {list(messy_df.columns)}")
    print(f"  Missing values:\n{messy_df.isnull().sum().to_string()}")

    question = "What is the total revenue by region, and which product sells the most units?"
    print(f"\n[Question] {question}")

    print("\n[Pipeline] Starting multi-agent analysis...")
    print("-" * 60)

    orchestrator = DataScientistOrchestrator()
    result = orchestrator.run(df=messy_df, question=question)

    print("-" * 60)
    print("\n[Results]")
    print(f"  Success: {result.success}")
    print(f"  Cleaned shape: {result.cleaned_df.shape if result.cleaned_df is not None else 'N/A'}")

    if result.cleaned_df is not None:
        print("\n  Cleaned data preview:")
        print(result.cleaned_df.head().to_string(index=False))

    if result.analysis_output:
        print("\n  Analysis output:")
        print(result.analysis_output)

    if result.chart_paths:
        print(f"\n  Charts saved: {result.chart_paths}")

    if result.summary:
        print("\n  Summary:")
        print(result.summary)

    if result.errors:
        print("\n  Errors:")
        for err in result.errors:
            print(f"    - {err}")

    print("\n[Progress Log]")
    for step in result.progress:
        print(f"  {step}")

    print("\n" + "=" * 60)
    print("Pipeline test complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
