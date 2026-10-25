"""Render docs/assets/bench.png from a measured `peftquant bench --json FILE` result.

Usage: python scripts/plot_bench.py [docs/assets/bench.json] [docs/assets/bench.png]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def plot(data: dict, out: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    w = data["weight_sqnr"]
    axes[0].scatter([r["ratio_vs_fp32"] for r in w], [r["sqnr_db"] for r in w])
    for r in w:
        axes[0].annotate(
            r["spec"], (r["ratio_vs_fp32"], r["sqnr_db"]), fontsize=7, xytext=(3, 3), textcoords="offset points"
        )
    axes[0].set_xlabel("compression vs fp32 (x)")
    axes[0].set_ylabel("SQNR dB (higher is better)")
    axes[0].set_title("Weight quantisation, randn 256x1024")
    rows = data["model_quant"]["rows"]
    specs = list(dict.fromkeys(r["spec"] for r in rows))
    get = {(r["spec"], r["method"]): r["logit_rel_err"] for r in rows}
    x = range(len(specs))
    axes[1].bar([i - 0.2 for i in x], [get[(s, "rtn")] for s in specs], 0.4, label="RTN")
    axes[1].bar([i + 0.2 for i in x], [get[(s, "gptq")] for s in specs], 0.4, label="GPTQ")
    axes[1].set_xticks(list(x), [s.replace(" ", "\n") for s in specs], fontsize=8)
    axes[1].set_ylabel("logit relative error vs fp32")
    axes[1].set_title("PTQ of trained MLP: GPTQ vs RTN")
    axes[1].legend()
    fig.suptitle("peftquant bench — seed 0, CPU (measured)")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out


if __name__ == "__main__":
    src = Path(sys.argv[1] if len(sys.argv) > 1 else "docs/assets/bench.json")
    dst = Path(sys.argv[2] if len(sys.argv) > 2 else "docs/assets/bench.png")
    print(plot(json.loads(src.read_text()), dst))
