"""
analysis/plot_speed.py  —  Post-Hoc Trace Analysis & Plotting
════════════════════════════════════════════════════════════════
Queries `logs/negotiation_log.sqlite` to generate:
  1. Negotiation speed (rounds-to-agreement) pre vs post price shock
  2. Price trajectory across negotiation rounds per scenario
  3. Violated Horn-clause frequency breakdown chart

Output plots are saved into `analysis/plots/`.
════════════════════════════════════════════════════════════════
"""

import sqlite3
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

_DB_PATH = Path(__file__).parent.parent / "logs" / "negotiation_log.sqlite"
_OUTPUT_DIR = Path(__file__).parent / "plots"
_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def plot_price_trajectories() -> Path:
    """Plot price trajectory round by round across sessions."""
    if not _DB_PATH.exists():
        print(f"Database {_DB_PATH} does not exist yet. Run controller first.")
        return _OUTPUT_DIR

    con = sqlite3.connect(_DB_PATH)
    df = pd.read_sql_query("SELECT * FROM negotiation_rounds", con)
    con.close()

    if df.empty:
        print("No negotiation round data found.")
        return _OUTPUT_DIR

    fig, ax = plt.subplots(figsize=(10, 6))

    sessions = df["session_id"].unique()
    for sid in sessions:
        sub = df[df["session_id"] == sid].sort_values("round_num")
        ax.plot(
            sub["round_num"],
            sub["offer_price"],
            marker="o",
            linewidth=2,
            label=f"Session {sid}",
        )
        
        # Highlight shock round if present
        shock_sub = sub[sub["shock_round"] == 1]
        if not shock_sub.empty:
            ax.scatter(
                shock_sub["round_num"],
                shock_sub["offer_price"],
                color="red",
                s=150,
                zorder=5,
                marker="⚡",
            )

    ax.set_title("Price Trajectory per Negotiation Round", fontsize=14, fontweight="bold")
    ax.set_xlabel("Negotiation Round", fontsize=12)
    ax.set_ylabel("Offer Price ($)", fontsize=12)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(loc="best")

    output_file = _OUTPUT_DIR / "price_trajectories.png"
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()
    print(f"Saved: {output_file}")
    return output_file


def plot_shock_impact() -> Path:
    """Plot comparison of negotiation speed and price pre vs post shock."""
    if not _DB_PATH.exists():
        print(f"Database {_DB_PATH} does not exist yet.")
        return _OUTPUT_DIR

    con = sqlite3.connect(_DB_PATH)
    summary_df = pd.read_sql_query("SELECT * FROM episode_summary", con)
    con.close()

    if summary_df.empty:
        print("No summary data found.")
        return _OUTPUT_DIR

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Rounds comparison
    scenarios = summary_df["scenario"].unique()
    rounds = [summary_df[summary_df["scenario"] == s]["rounds_to_agreement"].mean() for s in scenarios]
    
    ax1.bar(scenarios, rounds, color=["#3498db", "#e74c3c", "#2ecc71", "#9b59b6", "#f1c40f"][:len(scenarios)])
    ax1.set_title("Average Rounds to Agreement", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Rounds")
    ax1.grid(axis="y", linestyle="--", alpha=0.7)

    # Status breakdown
    status_counts = summary_df.groupby(["scenario", "status"]).size().unstack(fill_value=0)
    status_counts.plot(kind="bar", stacked=True, ax=ax2, color=["#2ecc71", "#e74c3c"])
    ax2.set_title("Outcome by Scenario (SUCCESS vs FAILED)", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Episode Count")
    ax2.grid(axis="y", linestyle="--", alpha=0.7)

    output_file = _OUTPUT_DIR / "shock_impact_analysis.png"
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()
    print(f"Saved: {output_file}")
    return output_file


def plot_violated_clauses() -> Path:
    """Plot frequency of each Horn clause violation."""
    if not _DB_PATH.exists():
        return _OUTPUT_DIR

    con = sqlite3.connect(_DB_PATH)
    df = pd.read_sql_query("SELECT violated_clauses FROM negotiation_rounds WHERE verdict = 'REJECTED'", con)
    con.close()

    if df.empty:
        print("No rejected rounds to plot clauses for.")
        return _OUTPUT_DIR

    clause_counts: dict[str, int] = {}
    for raw in df["violated_clauses"].dropna():
        for clause in raw.split(","):
            clause = clause.strip()
            if clause:
                clause_counts[clause] = clause_counts.get(clause, 0) + 1

    if not clause_counts:
        return _OUTPUT_DIR

    fig, ax = plt.subplots(figsize=(8, 5))
    clauses = list(clause_counts.keys())
    counts = list(clause_counts.values())

    ax.barh(clauses, counts, color="#e67e22")
    ax.set_title("Horn-Clause Violation Frequency", fontsize=12, fontweight="bold")
    ax.set_xlabel("Rejection Count")
    ax.grid(axis="x", linestyle="--", alpha=0.7)

    output_file = _OUTPUT_DIR / "violated_clauses.png"
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()
    print(f"Saved: {output_file}")
    return output_file


if __name__ == "__main__":
    print("Generating post-hoc negotiation analysis plots...")
    plot_price_trajectories()
    plot_shock_impact()
    plot_violated_clauses()
    print("Plots generated successfully in analysis/plots/")
