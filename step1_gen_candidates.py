#!/usr/bin/env python3
"""
Step 1 (GPU box): sample K candidate solutions per OJBench problem.

The problem prompt is sent VERBATIM -- no system prompt, no extra instructions.
That is what makes the baseline comparable to the number in the technical report.

    python step1_gen_candidates.py --config config.yaml
    python step1_gen_candidates.py --config config.yaml --limit 10 --K 4   # smoke test
    python step1_gen_candidates.py --config config.yaml --mode generator   # input generators

Output: <out_dir>/candidates.jsonl, one line per (problem, sample_idx).
Safe to re-run: completed samples are skipped.
"""
from __future__ import annotations

import argparse
import asyncio
import random
from pathlib import Path

from common import (JsonlAppender, done_keys, extract_code, load_config,
                    read_jsonl)
from llm import Progress, VLLMClient, wait_for_server

GENERATOR_PROMPT = """You are given a competitive programming problem statement.

Write a Python 3 script that prints ONE random VALID input for this problem to \
stdout. The script must:
- read an integer seed from sys.argv[1] and use random.seed(seed)
- respect every constraint in the statement (ranges, sum limits, graph validity, \
sortedness, and so on)
- generate a SMALL case: keep every size parameter at most 8, and every value \
small enough to check by hand
- print the input in exactly the format the problem specifies, nothing else

Reply with a single ```python code block and no other text.

--- PROBLEM ---
{statement}
"""


def select_problems(rows: list[dict], cfg: dict, limit: int | None,
                    seed: int) -> list[dict]:
    lang = cfg["data"]["language"]
    rows = [r for r in rows if r.get("language") == lang]
    if cfg["data"]["dataset"] != "all":
        rows = [r for r in rows if r.get("dataset") == cfg["data"]["dataset"]]
    if cfg["data"]["difficulty"] != "all":
        rows = [r for r in rows if r.get("difficulty") == cfg["data"]["difficulty"]]
    rows.sort(key=lambda r: r["id"])
    if limit:
        # Stratify the smoke-test subset across difficulties so it is informative.
        by_diff: dict[str, list[dict]] = {}
        for r in rows:
            by_diff.setdefault(r.get("difficulty", "?"), []).append(r)
        rng = random.Random(seed)
        picked: list[dict] = []
        per = max(1, limit // max(1, len(by_diff)))
        for _, group in sorted(by_diff.items()):
            picked.extend(rng.sample(group, min(per, len(group))))
        for r in rows:
            if len(picked) >= limit:
                break
            if r not in picked:
                picked.append(r)
        rows = sorted(picked, key=lambda r: r["id"])[:limit]
    return rows


async def gen_solutions(cfg: dict, rows: list[dict], out_path: Path, K: int) -> None:
    s = cfg["sampling"]
    lang = cfg["data"]["language"]
    client = VLLMClient(cfg["model"]["base_url"], cfg["model"]["served_name"],
                        cfg["model"]["api_key"], cfg["run"]["concurrency"])
    have = done_keys(out_path, ("id", "sample_idx"))

    # Build the list of (row, start_idx, n) request units.
    units: list[tuple[dict, int, int]] = []
    per_req = int(s["n_per_request"])
    for r in rows:
        missing = [i for i in range(K) if (r["id"], i) not in have]
        for i in range(0, len(missing), per_req):
            chunk = missing[i:i + per_req]
            units.append((r, chunk[0], len(chunk)))

    if not units:
        print("nothing to do -- all candidates already generated")
        await client.close()
        return

    total = sum(u[2] for u in units)
    prog = Progress(total, "generate")
    writer = JsonlAppender(out_path)
    lock = asyncio.Lock()

    async def one(row: dict, start: int, n: int) -> None:
        try:
            comps = await client.chat(
                row["prompt"], n=n,
                temperature=s["temperature"], top_p=s["top_p"],
                top_k=s["top_k"], max_tokens=s["max_tokens"],
            )
        except Exception as e:  # noqa: BLE001
            print(f"\n[warn] id={row['id']} failed: {e}")
            return
        async with lock:
            for j, c in enumerate(comps):
                code = extract_code(c.text, lang)
                writer.write({
                    "id": row["id"],
                    "sample_idx": start + j,
                    "dataset": row.get("dataset"),
                    "language": lang,
                    "difficulty": row.get("difficulty"),
                    "content": c.text,
                    "code": code,
                    "has_code": code is not None,
                    "finish_reason": c.finish_reason,
                    "completion_tokens": c.completion_tokens,
                })
                prog.tick()

    await asyncio.gather(*(one(*u) for u in units))
    prog.close()
    writer.close()
    await client.close()

    rows_out = read_jsonl(out_path)
    trunc = sum(1 for r in rows_out if r.get("finish_reason") == "length")
    nocode = sum(1 for r in rows_out if not r.get("has_code"))
    print(f"\nwrote {len(rows_out)} candidates to {out_path}")
    print(f"  truncated by max_tokens : {trunc} ({100 * trunc / max(1, len(rows_out)):.1f}%)")
    print(f"  no code block found     : {nocode} ({100 * nocode / max(1, len(rows_out)):.1f}%)")
    if trunc > 0.15 * len(rows_out):
        print("  ^ raise sampling.max_tokens; truncated traces almost never compile")


async def gen_generators(cfg: dict, rows: list[dict], out_path: Path) -> None:
    """One input-generator script per problem, used later for behavioural clustering."""
    c = cfg["clr_sampling"]
    client = VLLMClient(cfg["model"]["base_url"], cfg["model"]["served_name"],
                        cfg["model"]["api_key"], cfg["run"]["concurrency"])
    have = done_keys(out_path, ("id",))
    todo = [r for r in rows if r["id"] not in have]
    if not todo:
        print("all generators already present")
        await client.close()
        return

    prog = Progress(len(todo), "generators")
    writer = JsonlAppender(out_path)
    lock = asyncio.Lock()

    async def one(row: dict) -> None:
        try:
            comps = await client.chat(
                GENERATOR_PROMPT.format(statement=row["prompt"]),
                n=1, temperature=c["temperature"], top_p=c["top_p"],
                max_tokens=c["max_tokens_extract"],
            )
            gen_code = extract_code(comps[0].text, "python")
        except Exception as e:  # noqa: BLE001
            print(f"\n[warn] generator id={row['id']}: {e}")
            gen_code = None
        async with lock:
            writer.write({"id": row["id"], "generator_code": gen_code})
            prog.tick()

    await asyncio.gather(*(one(r) for r in todo))
    prog.close()
    writer.close()
    await client.close()
    print(f"\nwrote generators to {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--mode", choices=["solutions", "generator"], default="solutions")
    ap.add_argument("--limit", type=int, default=None,
                    help="only use N problems (stratified) -- for smoke tests")
    ap.add_argument("--K", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = load_config(args.config)
    K = args.K or cfg["run"]["K"]
    out_dir = Path(cfg["run"]["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    prompts_path = cfg["data"]["prompts"]
    if not Path(prompts_path).exists():
        raise SystemExit(f"prompts file not found: {prompts_path}\n"
                         "Download it from https://huggingface.co/datasets/He-Ren/OJBench_testdata")
    rows = select_problems(read_jsonl(prompts_path), cfg, args.limit, args.seed)
    print(f"{len(rows)} problems | language={cfg['data']['language']} | K={K}")

    asyncio.run(wait_for_server(cfg["model"]["base_url"]))

    if args.mode == "solutions":
        asyncio.run(gen_solutions(cfg, rows, out_dir / "candidates.jsonl", K))
    else:
        asyncio.run(gen_generators(cfg, rows, out_dir / "generators.jsonl"))


if __name__ == "__main__":
    main()
