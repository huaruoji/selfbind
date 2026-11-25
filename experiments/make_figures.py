#!/usr/bin/env python3
"""Generate the two data figures used by the SelfBind paper.

The script reads all plotted values from the released JSON artifacts and writes
PDF and PNG versions to ``paper/figures``.  It never edits the paper source.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent.parent
FIGURE_DIR = ROOT / "paper" / "figures"

RELEVANT_COLOR = "#2878B5"
IRRELEVANT_COLOR = "#D9534F"
GRID_COLOR = "#D8D8D8"


def configure_matplotlib() -> None:
    """Use compact, print-friendly defaults suitable for ACL figures."""
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
            "font.size": 7.5,
            "axes.labelsize": 8,
            "axes.titlesize": 8.5,
            "axes.linewidth": 0.65,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def load_gate_data(path: Path) -> dict[str, object]:
    """Load the evaluation-only gate records and derive decision counts."""
    data = json.loads(path.read_text())
    rows = [row for row in data["rows"] if row.get("gate_split") != "gate_train"]
    if not rows:
        raise ValueError(f"No evaluation-only rows in {path}")

    threshold = float(data["gate_metrics"]["threshold"])
    relevant = [row for row in rows if int(row["scope_label"]) == 1]
    irrelevant = [row for row in rows if int(row["scope_label"]) == 0]
    false_negatives = sum(
        float(row["gate_probability"]) < threshold for row in relevant
    )
    false_positives = sum(
        float(row["gate_probability"]) >= threshold for row in irrelevant
    )
    return {
        "threshold": threshold,
        "relevant": relevant,
        "irrelevant": irrelevant,
        "false_negatives": false_negatives,
        "false_positives": false_positives,
    }


def deterministic_jitter(rows: list[dict[str, object]], width: float = 0.19) -> np.ndarray:
    """Return stable horizontal jitter with no implied ordering of records."""
    offsets = []
    for row in rows:
        key = str(row.get("record_id", "")).encode("utf-8")
        value = int.from_bytes(hashlib.sha256(key).digest()[:8], "big")
        offsets.append(((value / (2**64 - 1)) - 0.5) * 2 * width)
    return np.asarray(offsets)


def make_gate_distribution() -> None:
    datasets = [
        (
            "Qwen3-4B-Instruct",
            load_gate_data(
                ROOT
                / "data/selfbind/benchmark_v2_qwen_aug_s42_rerun.json"
            ),
        ),
        (
            "LLaMA-3.1-8B-Instruct",
            load_gate_data(
                ROOT
                / "data/selfbind/benchmark_v2_llama_researchkb_s42_rerun.json"
            ),
        ),
    ]

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(7.0, 1.95),
        sharey=True,
        gridspec_kw={"wspace": 0.18},
    )
    legend_handles = None

    for ax, (model_name, gate) in zip(axes, datasets):
        irrelevant = gate["irrelevant"]
        relevant = gate["relevant"]
        threshold = float(gate["threshold"])

        irr_scatter = ax.scatter(
            deterministic_jitter(irrelevant),
            [float(row["gate_probability"]) for row in irrelevant],
            s=16,
            color=IRRELEVANT_COLOR,
            alpha=0.82,
            edgecolors="white",
            linewidths=0.25,
            zorder=3,
            label="Identity-irrelevant",
        )
        rel_scatter = ax.scatter(
            1 + deterministic_jitter(relevant),
            [float(row["gate_probability"]) for row in relevant],
            s=16,
            color=RELEVANT_COLOR,
            alpha=0.68,
            edgecolors="white",
            linewidths=0.2,
            zorder=3,
            label="Identity-relevant",
        )
        legend_handles = [rel_scatter, irr_scatter]

        ax.axhline(
            threshold,
            color="#222222",
            linestyle=(0, (4, 2.5)),
            linewidth=0.9,
            zorder=2,
        )
        threshold_label_y = threshold + (0.055 if threshold < 0.5 else -0.075)
        ax.text(
            0.5,
            threshold_label_y,
            rf"$\tau = {threshold:.2f}$",
            ha="center",
            va="center",
            fontsize=7,
            bbox={"boxstyle": "round,pad=0.16", "fc": "white", "ec": "none"},
            zorder=4,
        )
        ax.text(
            0.5,
            0.53,
            f"False positives: {gate['false_positives']}\n"
            f"False negatives: {gate['false_negatives']}",
            ha="center",
            va="center",
            fontsize=6.8,
            linespacing=1.25,
            color="#333333",
        )

        ax.set_title(model_name, pad=3)
        ax.set_xticks([0, 1], ["Identity-\nirrelevant", "Identity-\nrelevant"])
        ax.set_xlim(-0.42, 1.42)
        ax.set_ylim(-0.025, 1.045)
        ax.set_yticks(np.linspace(0, 1, 6))
        ax.set_axisbelow(True)
        ax.yaxis.grid(True, color=GRID_COLOR, linewidth=0.45, alpha=0.75)

    axes[0].set_ylabel(r"Gate probability $g(x)$")
    fig.legend(
        legend_handles,
        ["Identity-relevant", "Identity-irrelevant"],
        loc="upper center",
        bbox_to_anchor=(0.5, 1.015),
        ncol=2,
        frameon=False,
        handletextpad=0.45,
        columnspacing=1.6,
    )
    fig.subplots_adjust(left=0.075, right=0.995, bottom=0.23, top=0.76)
    save_figure(fig, "gate_distribution")


def parse_percentage(value: str) -> float:
    """Extract the terminal percentage from a ledger value such as 1213/1319=92.0%."""
    match = re.search(r"=([0-9]+(?:\.[0-9]+)?)%$", value)
    if match is None:
        raise ValueError(f"Cannot parse percentage from {value!r}")
    return float(match.group(1))


def load_utility_data() -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
    utility = json.loads((ROOT / "data/numbers_ledger.json").read_text())["utility"]
    methods = [
        ("Frozen\nbase", "base"),
        ("Localized\nq+v LoRA", "lora_qv"),
        ("MLP-only LoRA\n(DIVERSE)", "adapter_diverse"),
        ("Residual\npatch", "residual_patch"),
        ("SelfBind\nrouted", "selfbind_routed"),
    ]
    labels = [label for label, _ in methods]
    gsm8k = np.asarray(
        [parse_percentage(utility[key]["gsm8k"]) for _, key in methods]
    )
    mmlu = np.asarray(
        [parse_percentage(utility[key]["mmlu"]) for _, key in methods]
    )
    totals = np.asarray(
        [parse_percentage(utility[key]["total"]) for _, key in methods]
    )
    return labels, gsm8k, mmlu, totals


def make_utility_comparison() -> None:
    labels, gsm8k, mmlu, totals = load_utility_data()
    x = np.arange(len(labels))
    width = 0.29
    dark = ["#68777A"] * 4 + ["#2A8C62"]
    light = ["#B5BEC2"] * 4 + ["#78BE9A"]

    fig, ax = plt.subplots(figsize=(7.0, 2.18))
    ax.axvspan(3.55, 4.45, color="#2A8C62", alpha=0.07, zorder=0)
    ax.bar(
        x - width / 2,
        gsm8k,
        width,
        label="GSM8K (1,319)",
        color=dark,
        edgecolor="#333333",
        linewidth=0.45,
        zorder=2,
    )
    ax.bar(
        x + width / 2,
        mmlu,
        width,
        label="MMLU (3,000)",
        color=light,
        edgecolor="#333333",
        linewidth=0.45,
        zorder=2,
    )

    for x_pos, total in zip(x, totals):
        ax.text(
            x_pos,
            5.5,
            f"Overall {total:.1f}%",
            ha="center",
            va="center",
            fontsize=6.5,
            color="#222222",
            bbox={"boxstyle": "round,pad=0.18", "fc": "white", "ec": "none", "alpha": 0.94},
            zorder=3,
        )

    ax.text(
        4,
        97.0,
        "Gate closed = frozen base",
        ha="center",
        va="center",
        fontsize=7,
        color="#1F704D",
    )
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 102)
    ax.set_xticks(x, labels)
    ax.set_yticks(np.arange(0, 101, 20))
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color=GRID_COLOR, linewidth=0.45, alpha=0.8)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.16),
        ncol=2,
        frameon=False,
        handletextpad=0.45,
        columnspacing=1.6,
    )
    fig.subplots_adjust(left=0.075, right=0.995, bottom=0.23, top=0.84)
    save_figure(fig, "utility_comparison")


def save_figure(fig: plt.Figure, stem: str) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_DIR / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(
        FIGURE_DIR / f"{stem}.png",
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.02,
    )
    plt.close(fig)


def main() -> None:
    configure_matplotlib()
    make_gate_distribution()
    make_utility_comparison()
    for stem in ("gate_distribution", "utility_comparison"):
        for suffix in ("pdf", "png"):
            path = FIGURE_DIR / f"{stem}.{suffix}"
            print(f"{path.relative_to(ROOT)} ({path.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
