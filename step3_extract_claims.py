#!/usr/bin/env python3
"""
Step 3 (GPU box): CLR stage one -- pull M decision-relevant claims out of each
candidate.

The report's CLR extracts free-form claims from the trajectory. For code we fix
the five claim SLOTS instead. A 3B model emits a rigid five-line template far
more reliably than free-form JSON, and the slots map onto the five ways a
competitive-programming submission actually dies: wrong approach, too slow,
subtly wrong logic, unhandled edge case, wrong I/O format.

    python step3_extract_claims.py --config config.yaml

Output: <out_dir>/claims.jsonl
"""
from __future__ import annotations

import argparse
import asyncio
import re
from pathlib import Path

from common import (JsonlAppender, done_keys, iter_jsonl, load_config,
                    read_jsonl)
from llm import Progress, VLLMClient, wait_for_server

SLOTS = ["approach", "complexity", "correctness", "edge_cases", "io_format"]

EXTRACT_PROMPT = """Below is a competitive programming problem and a candidate \
solution. State the five load-bearing assumptions this specific solution depends \
on. Each must be a concrete, checkable factual statement about THIS code and THIS \
problem -- not a description of what the code does.

Bad:  "The code uses a segment tree."
Good: "A segment tree over the value axis answers each query in O(log V), so the \
total is O(n log V) which fits n <= 2*10^5 within the 1 second limit."

Reply in EXACTLY this format, one line each, nothing before or after:

CLAIM 1 (approach): <the reduction or algorithmic idea, and why it solves the \
stated problem>
CLAIM 2 (complexity): <time and memory complexity, with the actual constraint \
values, and whether they fit the stated limits>
CLAIM 3 (correctness): <the key invariant, recurrence or greedy exchange argument \
this code relies on being true>
CLAIM 4 (edge_cases): <the boundary conditions this code handles, e.g. minimum \
sizes, ties, empty results, integer overflow>
CLAIM 5 (io_format): <exactly what the code reads and exactly what it prints, and \
that this matches the required format>

--- PROBLEM ---
{statement}

--- CANDIDATE SOLUTION ({language}) ---
```{language}
{code}
```
"""

CLAIM_RE = re.compile(
    r"^\s*CLAIM\s*(\d)\s*\(([a-z_]+)\)\s*:\s*(.+?)\s*$",
    re.IGNORECASE | re.MULTILINE,
)


def parse_claims(text: str, m: int) -> list[dict]:
    """Extract the claim lines. Returns [] if fewer than 2 parse (unusable)."""
    found: dict[int, dict] = {}
    for num, slot, body in CLAIM_RE.findall(text or ""):
        i = int(num)
        if 1 <= i <= m and len(body.strip()) > 10:
            found[i] = {"index": i,
                        "slot": slot.lower(),
                        "text": body.strip()[:1200]}
    claims = [found[i] for i in sorted(found)]
    return claims if len(claims) >= 2 else []


async def run(cfg: dict, only_compiling: bool) -> None:
    out_dir = Path(cfg["run"]["out_dir"])
    cand_path = out_dir / "candidates.jsonl"
    if not cand_path.exists():
        raise SystemExit(f"missing {cand_path} -- run step1 first")

    stmts = {r["id"]: r["prompt"] for r in read_jsonl(cfg["data"]["prompts"])
             if r.get("language") == cfg["data"]["language"]}

    keep: set[tuple[int, int]] | None = None
    lt_path = out_dir / "local_tests.jsonl"
    if only_compiling:
        if not lt_path.exists():
            raise SystemExit("--only-compiling needs local_tests.jsonl -- run step2 "
                             "first, or pass --all")
        keep = {(r["id"], r["sample_idx"]) for r in iter_jsonl(lt_path) if r["compiles"]}
        print(f"restricting to {len(keep)} compiling candidates")

    M = cfg["run"]["M"]
    lang = cfg["data"]["language"]
    out_path = out_dir / "claims.jsonl"
    have = done_keys(out_path, ("id", "sample_idx"))

    todo = []
    for row in iter_jsonl(cand_path):
        k = (row["id"], row["sample_idx"])
        if k in have or not row.get("code"):
            continue
        if keep is not None and k not in keep:
            continue
        todo.append(row)

    if not todo:
        print("nothing to do")
        return

    c = cfg["clr_sampling"]
    client = VLLMClient(cfg["model"]["base_url"], cfg["model"]["served_name"],
                        cfg["model"]["api_key"], cfg["run"]["concurrency"])
    prog = Progress(len(todo), "claims")
    writer = JsonlAppender(out_path)
    lock = asyncio.Lock()

    async def one(row: dict) -> None:
        prompt = EXTRACT_PROMPT.format(
            statement=stmts.get(row["id"], ""),
            language=lang,
            code=row["code"][:20000],
        )
        claims: list[dict] = []
        raw = ""
        try:
            comps = await client.chat(prompt, n=1, temperature=c["temperature"],
                                      top_p=c["top_p"],
                                      max_tokens=c["max_tokens_extract"])
            raw = comps[0].text
            claims = parse_claims(raw, M)
        except Exception as e:  # noqa: BLE001
            print(f"\n[warn] claims id={row['id']}#{row['sample_idx']}: {e}")
        async with lock:
            writer.write({
                "id": row["id"],
                "sample_idx": row["sample_idx"],
                "claims": claims,
                "n_claims": len(claims),
                "raw_tail": raw[-600:],
            })
            prog.tick()

    await asyncio.gather(*(one(r) for r in todo))
    prog.close()
    writer.close()
    await client.close()

    rows = read_jsonl(out_path)
    bad = sum(1 for r in rows if r["n_claims"] == 0)
    print(f"\nwrote {len(rows)} claim sets to {out_path}")
    print(f"  unparseable: {bad} ({100 * bad / max(1, len(rows)):.1f}%)")
    if bad > 0.2 * len(rows):
        print("  ^ high. Inspect raw_tail on a few rows; the model may be "
              "wandering off-format. Lowering clr_sampling.temperature helps.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--all", action="store_true",
                    help="also extract claims for candidates that do not compile")
    args = ap.parse_args()
    cfg = load_config(args.config)
    asyncio.run(wait_for_server(cfg["model"]["base_url"]))
    asyncio.run(run(cfg, only_compiling=not args.all))


if __name__ == "__main__":
    main()
