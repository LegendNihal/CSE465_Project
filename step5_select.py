#!/usr/bin/env python3
"""
Step 5 (GPU box, CPU only): turn K candidates per problem into ONE submission.

CLR for code, adapted from the math version in the report:

  r_k       = (mean verdict over that candidate's claims) ** (number of claims)
  score_k   = r_k**w_clr * (sample_pass_frac + eps)**w_samples
  cluster   = candidates sharing a behavioural signature (identical output on
              the sample tests and the generated inputs) -- this replaces the
              "cluster answers by equivalence" step, which cannot be done
              syntactically for programs
  winner    = highest-scoring member of the highest-scoring cluster

Also writes ablation variants so you can attribute any gain to the right stage.

    python step5_select.py --config config.yaml

Output: <out_dir>/submissions/*.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from common import iter_jsonl, load_config, write_jsonl

VARIANTS = ["random", "compile", "samples", "cluster", "clr"]


def load_all(out_dir: Path) -> dict[tuple[int, int], dict]:
    rec: dict[tuple[int, int], dict] = {}
    for r in iter_jsonl(out_dir / "candidates.jsonl"):
        rec[(r["id"], r["sample_idx"])] = {
            "id": r["id"], "sample_idx": r["sample_idx"],
            "dataset": r.get("dataset"), "language": r.get("language"),
            "difficulty": r.get("difficulty"),
            "content": r.get("content", ""), "code": r.get("code"),
            "compiles": False, "sample_pass_frac": None, "signature": None,
            "verdicts": [],
        }
    lt = out_dir / "local_tests.jsonl"
    if lt.exists():
        for r in iter_jsonl(lt):
            k = (r["id"], r["sample_idx"])
            if k in rec:
                rec[k]["compiles"] = r["compiles"]
                rec[k]["sample_pass_frac"] = r["sample_pass_frac"]
                rec[k]["signature"] = r["signature"]
    vd = out_dir / "verdicts.jsonl"
    if vd.exists():
        for r in iter_jsonl(vd):
            k = (r["id"], r["sample_idx"])
            if k in rec:
                rec[k]["verdicts"].append(bool(r["verdict"]))
    return rec


def reliability(verdicts: list[bool]) -> float:
    """r = (mean v) ** m, the report's nonlinear penalty. No claims -> neutral."""
    if not verdicts:
        return 0.5
    m = len(verdicts)
    return (sum(verdicts) / m) ** m


def score_of(c: dict, cfg: dict) -> float:
    w_clr = cfg["select"]["w_clr"]
    w_s = cfg["select"]["w_samples"]
    eps = cfg["select"]["eps"]
    r = reliability(c["verdicts"])
    frac = c["sample_pass_frac"]
    s_term = 1.0 if frac is None else (frac + eps) ** w_s
    return (r ** w_clr) * s_term


def eligible(cands: list[dict], cfg: dict) -> list[dict]:
    """Apply hard gates, backing off so we always submit something."""
    pool = cands
    if cfg["select"]["hard_gate_compile"]:
        c2 = [c for c in pool if c["compiles"]]
        if c2:
            pool = c2
    if cfg["select"]["hard_gate_samples"]:
        c3 = [c for c in pool if c["sample_pass_frac"] in (None, 1.0)]
        if c3:
            pool = c3
    return pool


def pick(cands: list[dict], variant: str, cfg: dict, rng: random.Random) -> dict:
    if variant == "random":
        return rng.choice(cands)

    if variant == "compile":
        pool = [c for c in cands if c["compiles"]] or cands
        return rng.choice(pool)

    if variant == "samples":
        pool = [c for c in cands if c["compiles"]] or cands
        best = max((c["sample_pass_frac"] or 0.0) for c in pool)
        pool = [c for c in pool if (c["sample_pass_frac"] or 0.0) == best]
        return rng.choice(pool)

    if variant == "cluster":
        pool = eligible(cands, cfg)
        groups = defaultdict(list)
        for i, c in enumerate(pool):
            groups[c["signature"] or f"__solo{i}"].append(c)
        biggest = max(groups.values(), key=len)
        return rng.choice(biggest)

    # full CLR
    pool = eligible(cands, cfg)
    groups: dict[str, list[dict]] = defaultdict(list)
    for i, c in enumerate(pool):
        groups[c["signature"] or f"__solo{i}"].append(c)
    scored = {g: sum(score_of(c, cfg) for c in members)
              for g, members in groups.items()}
    best_group = max(scored, key=lambda g: (scored[g], len(groups[g])))
    members = groups[best_group]
    return max(members, key=lambda c: (
        score_of(c, cfg),
        c["sample_pass_frac"] or 0.0,
        -len(c["code"] or ""),
        -c["sample_idx"],
    ))


def to_submission(c: dict) -> dict:
    return {
        "id": c["id"], "dataset": c["dataset"], "language": c["language"],
        "difficulty": c["difficulty"], "content": c["content"],
        "_sample_idx": c["sample_idx"],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--splits", type=int, default=1,
                    help="partition the K candidates into R disjoint groups and "
                         "select independently in each -- gives a variance estimate")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = load_config(args.config)
    out_dir = Path(cfg["run"]["out_dir"])
    sub_dir = out_dir / "submissions"
    sub_dir.mkdir(parents=True, exist_ok=True)

    rec = load_all(out_dir)
    if not rec:
        raise SystemExit("no candidates found -- run step1 first")

    by_problem: dict[int, list[dict]] = defaultdict(list)
    for c in rec.values():
        by_problem[c["id"]].append(c)
    for v in by_problem.values():
        v.sort(key=lambda c: c["sample_idx"])

    rng = random.Random(args.seed)
    stats = defaultdict(lambda: defaultdict(int))

    for split in range(args.splits):
        for variant in VARIANTS:
            rows = []
            for pid in sorted(by_problem):
                cands = by_problem[pid]
                if args.splits > 1:
                    cands = [c for c in cands
                             if c["sample_idx"] % args.splits == split]
                    if not cands:
                        continue
                chosen = pick(cands, variant, cfg, rng)
                rows.append(to_submission(chosen))
                stats[variant]["n"] += 1
                stats[variant]["compiles"] += int(chosen["compiles"])
                stats[variant]["samples_ok"] += int(
                    chosen["sample_pass_frac"] in (None, 1.0))
            suffix = f"_split{split}" if args.splits > 1 else ""
            write_jsonl(sub_dir / f"submission_{variant}{suffix}.jsonl", rows)

    # Baseline: judge the first N candidates of every problem individually and
    # average -> this is mean Pass@1, the number the report compares against.
    bn = cfg["judge"]["baseline_n"]
    base_rows = []
    for pid in sorted(by_problem):
        for c in by_problem[pid][:bn]:
            base_rows.append(to_submission(c))
    write_jsonl(sub_dir / "baseline_all.jsonl", base_rows)

    # Everything, for the pass@K ceiling.
    all_rows = [to_submission(c) for pid in sorted(by_problem)
                for c in by_problem[pid]]
    write_jsonl(sub_dir / "all_candidates.jsonl", all_rows)

    print(f"problems: {len(by_problem)} | candidates: {len(rec)}")
    print(f"submissions written to {sub_dir}\n")
    print(f"{'variant':<10} {'n':>5} {'compiles':>9} {'samples_ok':>11}")
    for v in VARIANTS:
        s = stats[v]
        print(f"{v:<10} {s['n']:>5} {s['compiles']:>9} {s['samples_ok']:>11}")
    print(f"\nbaseline_all.jsonl : {len(base_rows)} rows "
          f"(first {bn} candidates x {len(by_problem)} problems)")
    print(f"all_candidates.jsonl: {len(all_rows)} rows (for pass@K)")

    diag = {
        "candidates": len(rec),
        "compile_rate": round(sum(c["compiles"] for c in rec.values()) / len(rec), 3),
        "have_verdicts": sum(1 for c in rec.values() if c["verdicts"]),
        "mean_reliability": round(
            sum(reliability(c["verdicts"]) for c in rec.values()) / len(rec), 3),
    }
    print("\n" + json.dumps(diag, indent=2))


if __name__ == "__main__":
    main()
