"""Render sweep results as the table that goes in the README."""

from __future__ import annotations

from collections.abc import Sequence

from .evaluation import RunResult


def lift(recall: float, baseline: float) -> float:
    """How much of the gap between the random baseline and perfect was closed."""
    room = 1.0 - baseline
    return (recall - baseline) / room if room > 1e-9 else 0.0


def render_markdown(runs: Sequence[RunResult]) -> str:
    header = (
        "| configuration | chunks | R@1 | random R@1 | R@5 | random R@5 | lift@5 | R@20 | MRR | "
        "out-of-corpus AUC | s |"
    )
    separator = "|" + "---|" * 11
    lines = [header, separator]
    for run in runs:
        lines.append(
            f"| `{run.config.name}` | {run.n_chunks} | {run.recall_at[1]:.3f} | "
            f"{run.baseline_random_at[1]:.3f} | {run.recall_at[5]:.3f} | "
            f"{run.baseline_random_at[5]:.3f} | "
            f"{lift(run.recall_at[5], run.baseline_random_at[5]):+.3f} | "
            f"{run.recall_at[20]:.3f} | "
            f"{run.mrr:.3f} | {run.out_of_corpus_auc:.3f} | {run.seconds:.1f} |"
        )
    return "\n".join(lines)


def render_by_area(runs: Sequence[RunResult]) -> str:
    areas = sorted({area for run in runs for area in run.recall_by_area})
    lines = ["| configuration | " + " | ".join(areas) + " |", "|" + "---|" * (len(areas) + 1)]
    for run in runs:
        cells = " | ".join(f"{run.recall_by_area.get(area, float('nan')):.2f}" for area in areas)
        lines.append(f"| `{run.config.name}` | {cells} |")
    return "\n".join(lines)


def render_console(run: RunResult) -> str:
    return (
        f"{run.config.name:<32} chunks={run.n_chunks:<6} "
        f"R@1={run.recall_at[1]:.3f} R@5={run.recall_at[5]:.3f} "
        f"R@20={run.recall_at[20]:.3f} MRR={run.mrr:.3f} "
        f"AUC={run.out_of_corpus_auc:.3f} {run.seconds:.1f}s"
    )
