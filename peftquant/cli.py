"""``peftquant`` command line: ``bench``, ``grpo``, ``device``."""

from __future__ import annotations

import argparse
import json
import logging
import sys

from peftquant.benchmarks import BenchConfig, run_all
from peftquant.device import detect_device
from peftquant.grpo import GRPOConfig, train_grpo


def _fmt_bench(r: dict) -> str:
    lines = [
        "## Weight quantisation (randn 256x1024, fp32 = 1,048,576 B)",
        "| spec | SQNR dB | bytes | ratio |",
        "|---|---|---|---|",
    ]
    lines += [
        f"| {x['spec']} | {x['sqnr_db']:.2f} | {x['bytes']} | {x['ratio_vs_fp32']:.2f}x |" for x in r["weight_sqnr"]
    ]
    mq = r["model_quant"]
    lines += [
        "",
        f"## MLP post-training quantisation (fp32 acc {mq['fp32']['acc']:.4f}, {mq['fp32']['bytes']} B)",
        "| spec | method | acc | bytes | logit rel err | top-1 agree |",
        "|---|---|---|---|---|---|",
    ]
    lines += [
        f"| {x['spec']} | {x['method']} | {x['acc']:.4f} | {x['bytes']} "
        f"| {x['logit_rel_err']:.4f} | {x['top1_agree']:.4f} |"
        for x in mq["rows"]
    ]
    lines += [
        "",
        "## LoRA -> merge -> GPTQ -> export",
        *(f"- {k}: {v}" for k, v in r["lora_pipeline"].items()),
    ]
    lines += [
        "",
        "## Freivalds",
        *(f"- {k}: {v}" for k, v in r["freivalds_speed"].items()),
    ]
    lines += ["", "## GRPO toy", *(f"- {k}: {v}" for k, v in r["grpo"].items())]
    lines += [
        "",
        f"total {r['total_s']:.1f} s, torch {r['torch']}, {r['threads']} threads",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="peftquant", description=__doc__)
    ap.add_argument("-v", "--verbose", action="store_true")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("bench", help="run all CPU benchmarks")
    b.add_argument("--seed", type=int, default=0)
    b.add_argument("--json", help="also write raw results to this file")
    g = sub.add_parser("grpo", help="train the toy GRPO+LoRA policy")
    g.add_argument("--steps", type=int, default=400)
    g.add_argument("--kl", type=float, default=0.0)
    g.add_argument("--seed", type=int, default=0)
    sub.add_parser("device", help="print detected accelerator")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if a.verbose else logging.WARNING)
    if a.cmd == "bench":
        r = run_all(BenchConfig(seed=a.seed))
        if a.json:
            with open(a.json, "w") as fh:
                json.dump(r, fh, indent=2)
        print(_fmt_bench(r))
    elif a.cmd == "grpo":
        r2 = train_grpo(GRPOConfig(steps=a.steps, kl_coef=a.kl, seed=a.seed))
        r2.pop("sampled_reward_curve")
        print(json.dumps(r2, indent=2))
    else:
        d = detect_device()
        print(f"{d.backend}: {d.name} ({d.device})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
