#!/usr/bin/env python3
"""
Step 2 (GPU box, CPU only): execution signals that cost no GPU time.

For every candidate:
  1. does it compile / parse?
  2. does it pass the sample tests printed in the statement?
  3. what does it print on a handful of generated inputs?  (behavioural signature)

None of this touches OJBench's hidden test data. Everything used here is
information a human contestant has before submitting.

    python step2_local_tests.py inspect-samples --config config.yaml --n 5
    python step2_local_tests.py run --config config.yaml

Output: <out_dir>/local_tests.jsonl
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from common import (JsonlAppender, compile_candidate, done_keys, iter_jsonl,
                    load_config, normalize_output, outputs_match, parse_samples,
                    read_jsonl, run_candidate, temp_workdir)

# ---------------------------------------------------------------- worker


def _eval_candidate(task: dict) -> dict:
    """Runs in a subprocess. Compiles once, then runs every input."""
    code = task["code"]
    lang = task["language"]
    out = {
        "id": task["id"],
        "sample_idx": task["sample_idx"],
        "compiles": False,
        "compile_error": "",
        "n_samples": len(task["samples"]),
        "samples_passed": 0,
        "sample_pass_frac": None,
        "sample_details": [],
        "signature": None,
        "gen_statuses": [],
    }
    if not code:
        out["compile_error"] = "no code block in response"
        return out

    wd = temp_workdir()
    try:
        comp = compile_candidate(code, lang, wd, task["compile_timeout_s"])
        out["compiles"] = comp.ok
        if not comp.ok:
            out["compile_error"] = comp.error[:1500]
            return out

        # --- sample tests -------------------------------------------------
        passed = 0
        for s in task["samples"]:
            r = run_candidate(comp, s["stdin"], task["run_timeout_s"], task["mem_mb"])
            ok = r.status == "ok" and outputs_match(r.stdout, s["expected"])
            passed += int(ok)
            out["sample_details"].append({
                "status": r.status,
                "ok": ok,
                "seconds": round(r.seconds, 3),
            })
        out["samples_passed"] = passed
        if task["samples"]:
            out["sample_pass_frac"] = passed / len(task["samples"])

        # --- behavioural signature ---------------------------------------
        sig_parts = []
        for s in task["samples"]:
            r = run_candidate(comp, s["stdin"], task["run_timeout_s"], task["mem_mb"])
            sig_parts.append(r.status + "|" + normalize_output(r.stdout))
        for inp in task["generated_inputs"]:
            r = run_candidate(comp, inp, task["run_timeout_s"], task["mem_mb"])
            out["gen_statuses"].append(r.status)
            sig_parts.append(r.status + "|" + normalize_output(r.stdout))
        if sig_parts:
            blob = "\x1e".join(sig_parts).encode("utf-8", "replace")
            out["signature"] = hashlib.sha256(blob).hexdigest()[:20]
        return out
    finally:
        shutil.rmtree(wd, ignore_errors=True)


# ---------------------------------------------------------------- helpers


def build_generated_inputs(gen_code: str | None, n: int, timeout_s: float) -> list[str]:
    """Run the model-written generator n times with different seeds."""
    if not gen_code:
        return []
    wd = temp_workdir("gen_")
    try:
        src = Path(wd) / "gen.py"
        src.write_text(gen_code, encoding="utf-8")
        import subprocess
        inputs = []
        for seed in range(n):
            try:
                p = subprocess.run(["python3", str(src), str(seed)],
                                   capture_output=True, text=True,
                                   timeout=timeout_s, cwd=wd)
            except subprocess.TimeoutExpired:
                continue
            if p.returncode == 0 and p.stdout.strip():
                inputs.append(p.stdout)
        # Deduplicate; identical inputs add no discriminative power.
        seen, uniq = set(), []
        for i in inputs:
            if i not in seen:
                seen.add(i)
                uniq.append(i)
        return uniq
    finally:
        shutil.rmtree(wd, ignore_errors=True)


def load_statements(cfg: dict) -> dict[int, str]:
    return {r["id"]: r["prompt"] for r in read_jsonl(cfg["data"]["prompts"])
            if r.get("language") == cfg["data"]["language"]}


# ---------------------------------------------------------------- commands


def cmd_inspect(cfg: dict, n: int) -> None:
    """Print what the sample parser found. ALWAYS run this before a full run."""
    stmts = load_statements(cfg)
    ids = sorted(stmts)[:n]
    n_ok = 0
    for pid in ids:
        samples = parse_samples(stmts[pid])
        print("=" * 70)
        print(f"problem {pid}: parsed {len(samples)} sample test(s)")
        n_ok += bool(samples)
        for i, s in enumerate(samples):
            print(f"  --- sample {i} stdin ---")
            print("  " + s.stdin[:300].replace("\n", "\n  "))
            print(f"  --- sample {i} expected ---")
            print("  " + s.expected[:300].replace("\n", "\n  "))
    print("=" * 70)
    print(f"{n_ok}/{len(ids)} problems yielded at least one sample.")
    print("If this is below ~80%, open a statement by hand and widen the "
          "regexes in common.py:_IN_WORDS / _OUT_WORDS before continuing.")


def cmd_run(cfg: dict) -> None:
    out_dir = Path(cfg["run"]["out_dir"])
    cand_path = out_dir / "candidates.jsonl"
    if not cand_path.exists():
        raise SystemExit(f"missing {cand_path} -- run step1 first")

    lt = cfg["local_tests"]
    stmts = load_statements(cfg)

    # Sample tests, parsed once per problem.
    samples_by_id: dict[int, list[dict]] = {}
    for pid, text in stmts.items():
        samples_by_id[pid] = [{"stdin": s.stdin, "expected": s.expected}
                              for s in parse_samples(text)]

    # Generated inputs, built once per problem.
    gen_by_id: dict[int, list[str]] = {}
    gen_path = out_dir / "generators.jsonl"
    if lt["use_generated_inputs"] and gen_path.exists():
        print("building generated inputs ...")
        for row in iter_jsonl(gen_path):
            gen_by_id[row["id"]] = build_generated_inputs(
                row.get("generator_code"), lt["n_generated_inputs"],
                lt["run_timeout_s"])
        got = sum(1 for v in gen_by_id.values() if v)
        print(f"  usable generators: {got}/{len(gen_by_id)}")
    elif lt["use_generated_inputs"]:
        print("[note] no generators.jsonl -- run step1 --mode generator for "
              "stronger clustering. Continuing with sample tests only.")

    out_path = out_dir / "local_tests.jsonl"
    have = done_keys(out_path, ("id", "sample_idx"))

    tasks = []
    for row in iter_jsonl(cand_path):
        if (row["id"], row["sample_idx"]) in have:
            continue
        tasks.append({
            "id": row["id"],
            "sample_idx": row["sample_idx"],
            "code": row.get("code"),
            "language": row["language"],
            "samples": samples_by_id.get(row["id"], []),
            "generated_inputs": gen_by_id.get(row["id"], []),
            "run_timeout_s": lt["run_timeout_s"],
            "mem_mb": lt["mem_mb"],
            "compile_timeout_s": lt["compile_timeout_s"],
        })

    if not tasks:
        print("nothing to do")
        return

    print(f"evaluating {len(tasks)} candidates with {lt['workers']} workers")
    print("WARNING: this executes model-written code with only rlimit "
          "protection. Run it on the rented box, not your laptop.")

    done = 0
    with JsonlAppender(out_path) as writer, \
            ProcessPoolExecutor(max_workers=lt["workers"]) as pool:
        futs = [pool.submit(_eval_candidate, t) for t in tasks]
        for fut in as_completed(futs):
            writer.write(fut.result())
            done += 1
            if done % 25 == 0 or done == len(tasks):
                print(f"  {done}/{len(tasks)}", flush=True)

    rows = read_jsonl(out_path)
    n = len(rows)
    comp = sum(r["compiles"] for r in rows)
    with_s = [r for r in rows if r["sample_pass_frac"] is not None]
    full_s = sum(1 for r in with_s if r["sample_pass_frac"] == 1.0)
    print(json.dumps({
        "candidates": n,
        "compile_rate": round(comp / max(1, n), 3),
        "have_sample_tests": len(with_s),
        "pass_all_samples": full_s,
        "pass_all_samples_rate": round(full_s / max(1, len(with_s)), 3),
    }, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "inspect-samples"])
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--n", type=int, default=5)
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.cmd == "inspect-samples":
        cmd_inspect(cfg, args.n)
    else:
        cmd_run(cfg)


if __name__ == "__main__":
    main()
