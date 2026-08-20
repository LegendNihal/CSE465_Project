#!/usr/bin/env python3
"""
Step 4 (GPU box): CLR stage two -- the model tries to FALSIFY each of its own
claims, one claim per call, in a fresh context.

Two deliberate choices:
  * the original reasoning trace is NOT shown. Showing it makes the model agree
    with itself; the verdicts collapse to all-TRUE and carry no signal.
  * the framing is adversarial ("find a counterexample"), not "is this correct?".
    A 3B model says yes to almost anything phrased neutrally.

    python step4_verify_claims.py --config config.yaml

Output: <out_dir>/verdicts.jsonl
"""
from __future__ import annotations

import argparse
import asyncio
import re
from pathlib import Path

from common import (JsonlAppender, done_keys, iter_jsonl, load_config,
                    read_jsonl)
from llm import Progress, VLLMClient, wait_for_server

VERIFY_PROMPT = """You are reviewing a submission for a programming contest. \
Your job is to BREAK the claim below, not to agree with it.

Work through it concretely: construct the smallest input that would violate the \
claim, trace the code on it, or compute the actual operation count against the \
stated limits. If after a real attempt you cannot break it, the claim stands.

--- PROBLEM ---
{statement}

--- SUBMITTED CODE ({language}) ---
```{language}
{code}
```

--- CLAIM UNDER REVIEW ---
{claim}

Reason about the claim, then finish your reply with exactly one line, on its own:
VERDICT: TRUE
or
VERDICT: FALSE

TRUE means the claim holds for this code and this problem. FALSE means you found \
a concrete way it fails."""

VERDICT_RE = re.compile(r"VERDICT\s*:\s*(TRUE|FALSE)", re.IGNORECASE)


def parse_verdict(text: str) -> bool | None:
    """Last VERDICT line wins -- the model may restate it while reasoning."""
    hits = VERDICT_RE.findall(text or "")
    if not hits:
        return None
    return hits[-1].upper() == "TRUE"


async def run(cfg: dict) -> None:
    out_dir = Path(cfg["run"]["out_dir"])
    claims_path = out_dir / "claims.jsonl"
    if not claims_path.exists():
        raise SystemExit(f"missing {claims_path} -- run step3 first")

    stmts = {r["id"]: r["prompt"] for r in read_jsonl(cfg["data"]["prompts"])
             if r.get("language") == cfg["data"]["language"]}
    code_by_key = {(r["id"], r["sample_idx"]): r.get("code")
                   for r in iter_jsonl(out_dir / "candidates.jsonl")}

    lang = cfg["data"]["language"]
    c = cfg["clr_sampling"]
    unparsed_true = cfg["clr"]["unparsed_verdict_true"]
    retries = cfg["clr"]["verify_retries"]

    out_path = out_dir / "verdicts.jsonl"
    have = done_keys(out_path, ("id", "sample_idx", "claim_index"))

    todo = []
    for row in iter_jsonl(claims_path):
        code = code_by_key.get((row["id"], row["sample_idx"]))
        if not code:
            continue
        for cl in row["claims"]:
            if (row["id"], row["sample_idx"], cl["index"]) in have:
                continue
            todo.append((row["id"], row["sample_idx"], cl, code))

    if not todo:
        print("nothing to do")
        return

    print(f"{len(todo)} claim verifications queued")
    client = VLLMClient(cfg["model"]["base_url"], cfg["model"]["served_name"],
                        cfg["model"]["api_key"], cfg["run"]["concurrency"])
    prog = Progress(len(todo), "verify")
    writer = JsonlAppender(out_path)
    lock = asyncio.Lock()

    async def one(pid: int, sidx: int, claim: dict, code: str) -> None:
        prompt = VERIFY_PROMPT.format(
            statement=stmts.get(pid, ""), language=lang,
            code=code[:20000], claim=claim["text"],
        )
        verdict, parsed, tail = None, False, ""
        for _ in range(retries + 1):
            try:
                comps = await client.chat(prompt, n=1,
                                          temperature=c["temperature"],
                                          top_p=c["top_p"],
                                          max_tokens=c["max_tokens_verify"])
                tail = comps[0].text[-800:]
                v = parse_verdict(comps[0].text)
                if v is not None:
                    verdict, parsed = v, True
                    break
            except Exception as e:  # noqa: BLE001
                print(f"\n[warn] verify {pid}#{sidx}c{claim['index']}: {e}")
        if not parsed:
            verdict = unparsed_true
        async with lock:
            writer.write({
                "id": pid,
                "sample_idx": sidx,
                "claim_index": claim["index"],
                "slot": claim["slot"],
                "verdict": bool(verdict),
                "parsed": parsed,
                "raw_tail": tail,
            })
            prog.tick()

    await asyncio.gather(*(one(*t) for t in todo))
    prog.close()
    writer.close()
    await client.close()

    rows = read_jsonl(out_path)
    n = len(rows)
    unparsed = sum(1 for r in rows if not r["parsed"])
    true_rate = sum(r["verdict"] for r in rows) / max(1, n)
    print(f"\nwrote {n} verdicts to {out_path}")
    print(f"  unparsed  : {unparsed} ({100 * unparsed / max(1, n):.1f}%)")
    print(f"  TRUE rate : {true_rate:.3f}")
    if true_rate > 0.95:
        print("  ^ the verifier is rubber-stamping. CLR will add nothing at this "
              "rate. Try raising clr_sampling.temperature, or sharpen the "
              "adversarial wording in VERIFY_PROMPT.")
    by_slot: dict[str, list[int]] = {}
    for r in rows:
        by_slot.setdefault(r["slot"], []).append(int(r["verdict"]))
    print("  TRUE rate by slot:")
    for slot, vals in sorted(by_slot.items()):
        print(f"    {slot:<14} {sum(vals) / len(vals):.3f}  (n={len(vals)})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    asyncio.run(wait_for_server(cfg["model"]["base_url"]))
    asyncio.run(run(cfg))


if __name__ == "__main__":
    main()
