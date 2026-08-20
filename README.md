# CLR-Code: applying Claim-Level Reliability Assessment to VibeThinker-3B on OJBench

Adapting the CLR test-time scaling strategy from the VibeThinker-3B technical
report (arXiv 2606.16140) from answer-verifiable math to competitive programming,
measured on OJBench.

---

## 0. Read this before you spend money

**You are doing something the paper did not do.** In Table 2, the `+ CLR` row has
values for AIME25/26, HMMT25, BruMO25, IMO-AnswerBench and GPQA-Diamond. The
LCBv6 and OJBench columns are **blank**. CLR as published is defined for tasks
where the final answer is a short string you can cluster by exact equivalence.
A program is not that. So there is no published number to reproduce and no
guarantee this works — that is the point of running it.

**Three things make this harder than the math case:**

1. **Clustering doesn't transfer.** AIME answers are integers 0–999, so
   "cluster by equivalence" is a dictionary lookup and plain majority voting
   already recovers most of the gain. Two correct programs are almost never
   textually equal, and semantic equivalence is undecidable. This pipeline
   replaces it with *behavioural* clustering: run every candidate on the same
   inputs and group by output signature.
2. **The ceiling is low.** Selection can only recover the gap between mean
   pass@1 and pass@K. On NOI/ICPC-hard problems a 3B model often produces zero
   correct candidates in 16 tries, and no selector can fix that. **Measure your
   pass@K before building anything** — section 3, step 6.
3. **The verifier is the same 3B model.** If it can't solve the problem, it
   often can't falsify claims about it either. Watch the TRUE-rate diagnostic
   that `step4` prints; above ~0.95 the verifier is rubber-stamping and CLR is
   contributing nothing.

**Realistic expectation:** most of your gain will come from throwing away code
that doesn't compile or fails the sample tests — cheap, deterministic, and
nothing to do with CLR. The claim machinery is a tie-breaker on top of that.
That is a perfectly good result *provided you report the ablation* so the credit
lands in the right place. The pipeline produces that ablation for you.

---

## 1. Architecture

```
 VAST.AI (RTX 4090)                              WSL2 (Ubuntu 22.04)
 ─────────────────────────────                   ────────────────────────
 step1  generate K candidates      GPU
 step1  --mode generator           GPU
 step2  compile + sample tests     CPU
 step3  extract M claims           GPU
 step4  adversarially verify       GPU
 step5  aggregate → submissions    CPU
          │
          └── scp ~2 MB of jsonl ──────────────►  step6  DMOJ judge
                                                  step6  report
```

Everything except judging runs on the rented box. Two reasons:

- **Safety.** Steps 2 and 5 execute model-written C++. Rented hardware you
  destroy afterwards is the right place for that. The runner sets CPU, memory,
  process and file-size rlimits, but it is not a real sandbox — it does not
  block network or filesystem access.
- **The test data is big and the box is ephemeral.** `OJBench_testdata` is
  several GB of judge data. Download it once on WSL2 and keep it; don't re-pull
  it onto every new Vast instance.

Only small files cross the boundary — submission jsonl with one response per
problem.

---

## 2. Setup

### GPU box

Rent a 4090 with **≥60 GB disk**. Use the `vllm/vllm-openai:v0.10.1` image if
Vast offers it; the model card pins vLLM 0.10.1 and it saves a 10-minute install.

**Getting this code onto the box.** There is no repo to clone — copy the folder
up with `scp`. Vast shows an ssh command on the instance card like
`ssh -p 41822 root@ssh5.vast.ai`; reuse that port and host:

```bash
# from WSL2, where ~/clrcode holds these files
scp -P 41822 -r ~/clrcode root@ssh5.vast.ai:/workspace/
```

`ssh` takes lowercase `-p` for the port but `scp` takes uppercase `-P`. Copying
the port across with the wrong case is the usual reason this fails, and the
error message does not help.

The GPU box also needs the OJBench prompts file (only that one file, not the
test data):

```bash
ssh -p 41822 root@ssh5.vast.ai "mkdir -p /workspace/clrcode/OJBench_testdata/prompts"
scp -P 41822 ~/ojbench/OJBench_testdata/prompts/full.jsonl \
    root@ssh5.vast.ai:/workspace/clrcode/OJBench_testdata/prompts/
```

Once you start editing prompts you will want a private git repo instead, so
changes sync both ways. Then:

```bash
ssh -p 41822 root@ssh5.vast.ai
cd /workspace/clrcode
bash setup_vast.sh

tmux new -s vllm            # survives ssh drops — you will drop
bash serve_vllm.sh
# Ctrl-b then d to detach
```

Verify:

```bash
curl -s localhost:8000/v1/models | head -c 300
```

### WSL2 box

```bash
lsb_release -a              # must say 22.04
bash setup_wsl.sh
```

If you're on Ubuntu 24.04, install 22.04 alongside it (`wsl --install -d
Ubuntu-22.04`). DMOJ's judge-server is pinned by OJBench to a 2024 commit and
does not build against Python 3.12.

The two failure points people hit, in order of frequency:

| Symptom | Cause |
| --- | --- |
| `cptbox` fails to build | `libseccomp-dev` missing — `setup_wsl.sh` installs it |
| judging is glacially slow | testdata sits under `/mnt/c` — move it to `~` |
| `Operation not permitted` at judge time | `sudo sysctl -w kernel.unprivileged_userns_clone=1` |

---

## 3. Day 1 — get a real number in about four hours

Do this before scaling. It validates every interface and gives you the pass@K
headroom that determines whether the rest is worth doing.

**Start the testdata download first**, it runs unattended:

```bash
# WSL2, in its own terminal
cd ~/ojbench && git clone https://huggingface.co/datasets/He-Ren/OJBench_testdata
```

**1. Copy the prompts file to the GPU box** (it's small, you only need
`prompts/full.jsonl` there):

```bash
scp ~/ojbench/OJBench_testdata/prompts/full.jsonl \
    root@<vast-ip>:/workspace/clrcode/OJBench_testdata/prompts/
```

**2. Sanity-check the sample parser.** This is the step most likely to silently
break the whole pipeline, because it depends on markdown wording you haven't
seen:

```bash
python step2_local_tests.py inspect-samples --config config.yaml --n 8
```

Read the output. If fewer than ~80% of problems yield samples, open a statement
by hand and widen `_IN_WORDS` / `_OUT_WORDS` in `common.py`. **Do not skip
this** — with no sample tests, the strongest signal in the pipeline is gone and
CLR is running blind.

**3. Twelve problems, K=4:**

```bash
python step1_gen_candidates.py --config config.yaml --limit 12 --K 4
python step1_gen_candidates.py --config config.yaml --limit 12 --mode generator
python step2_local_tests.py   run --config config.yaml
python step3_extract_claims.py --config config.yaml
python step4_verify_claims.py  --config config.yaml
python step5_select.py         --config config.yaml
```

Each step prints diagnostics. Stop and fix if you see:

- step1: truncation rate above 15% → raise `sampling.max_tokens`
- step1: "no code block" above 10% → the prompts aren't reaching the model intact
- step3: unparseable above 20% → lower `clr_sampling.temperature`
- step4: TRUE rate above 0.95 → the verifier is rubber-stamping, see §6

**4. Ship it to WSL2 and judge:**

```bash
scp -r root@<vast-ip>:/workspace/clrcode/runs/cpp_k16/submissions ./
python step6_judge.py judge --config config.yaml --input submissions/all_candidates.jsonl
python step6_judge.py report --config config.yaml
```

**5. Read `pass@4` versus `mean pass@1` in the report.** That difference is your
entire budget. If pass@4 ≈ mean pass@1, selection has nothing to select from and
you should reconsider the project (or move to a track with more headroom — see
§7). If pass@4 is 10+ points above, you have room to work with.

---

## 4. Full run

Set `run.K: 16` and drop `--limit`:

```bash
tmux new -s pipeline
python step1_gen_candidates.py --config config.yaml
python step1_gen_candidates.py --config config.yaml --mode generator
python step2_local_tests.py   run --config config.yaml
python step3_extract_claims.py --config config.yaml
python step4_verify_claims.py  --config config.yaml
python step5_select.py         --config config.yaml --splits 2
```

Every step is resumable — re-running skips completed work by key. If your spot
instance dies at hour six, restart the same command.

Then on WSL2, judge the baseline, the ablations and the CLR submission:

```bash
python step6_judge.py judge --config config.yaml --input submissions/baseline_all.jsonl
for v in random compile samples cluster clr; do
  python step6_judge.py judge --config config.yaml --input submissions/submission_$v.jsonl
done
python step6_judge.py report --config config.yaml
```

### Budget, 232 problems × K=16, C++ track

| Stage | Work | 4090 wall-clock |
| --- | --- | --- |
| step1 solutions | 3,712 rollouts, ~12k tok each | 5–8 h |
| step1 generators | 232 short calls | ~15 min |
| step2 | CPU, 8 workers | 30–60 min |
| step3 claims | ~2,200 calls (compiling only) | ~30 min |
| step4 verify | ~11,000 calls × 5 claims | 2–3 h |
| step5 | seconds | — |

**~9–12 GPU-hours, roughly $4–6 at 4090 spot pricing.** Judging on WSL2 adds
1.5–3 hours of CPU for ~3,000 submissions.

Disk on the GPU box: model 6.2 GB + ~200 MB of generations. The 60 GB is
comfortable.

---

## 5. What the pipeline actually computes

**Reliability, straight from the report:**

```
r_k = (mean verdict over that candidate's claims) ** (number of claims)
```

With 5 claims, one FALSE gives `0.8^5 = 0.33` and two give `0.6^5 = 0.078`. The
exponent is what makes CLR heavily penalise flawed intermediate logic rather
than averaging it away.

**Candidate score:**

```
score_k = r_k**w_clr  ×  (sample_pass_fraction + eps)**w_samples
```

**Cluster score:** candidates producing byte-identical output on the sample
tests *and* the generated inputs share a signature; a cluster's score is the sum
of its members'. Pick the best cluster, then its best member. This is the
behavioural stand-in for the report's "cluster answers by equivalence" step.

Exact-match clustering is valid here because OJBench deliberately **excluded
problems requiring a special judge**, so a correct program's output is unique.

**Two design choices worth knowing about**, both in `step4_verify_claims.py`:

- The original reasoning trace is *not* shown to the verifier. Showing it makes
  the model agree with itself and the verdicts collapse to all-TRUE.
- The framing is adversarial ("break this claim, find a counterexample"), not
  neutral ("is this correct?"). A 3B model says yes to nearly anything phrased
  neutrally.

And the five claim slots — approach, complexity, correctness, edge cases, I/O
format — are fixed rather than free-form. A 3B model hits a rigid five-line
template far more reliably than free-form JSON (see the ZebraLogic thread on the
model's HF page for how badly it handles structured output), and the slots map
onto the five ways a contest submission actually dies.

---

## 6. If CLR adds nothing

The ablation table tells you where you are. Diagnose in this order:

**`clr` ≈ `cluster`** → the reliability scores aren't discriminating. Check
step4's TRUE rate. If it's above 0.95, the verifier rubber-stamps everything.
Try: raise `clr_sampling.temperature` to 0.8; sharpen `VERIFY_PROMPT` toward
demanding a concrete counterexample; or verify only the two slots with the
lowest TRUE rate (step4 prints per-slot rates — usually `complexity` and
`edge_cases` discriminate, `io_format` does not).

**`cluster` ≈ `samples`** → clustering isn't separating anything, which means
the generated inputs are too weak. Check `usable generators` in step2's output.
Raise `n_generated_inputs`, or edit `GENERATOR_PROMPT` to push for adversarial
rather than random inputs (maximum-size cases, all-equal values, degenerate
graphs).

**`samples` ≈ `compile`** → the sample parser is failing. Re-run
`inspect-samples`.

**Everything ≈ baseline, and pass@16 ≈ mean pass@1** → the model isn't producing
diverse-enough candidates. Nothing on the selection side can fix that. This is
the honest negative result, and it's worth writing up.

---

## 7. Integrity rules

**Never let OJBench's hidden tests influence selection.** Steps 1–5 must never
read from `OJBench_testdata/NOI` or `/ICPC`. Only step 6 does. The judge exposes
`1/8verdict` and `1/4verdict` fields that look tempting as a cheap signal —
using them to choose a submission makes the resulting number meaningless. Every
signal in this pipeline (statement samples, self-generated inputs, self-verified
claims) is information a human contestant has before they hit submit.

**Report your own baseline, not 38.6.** The paper's number comes from a
different sampling configuration and its language mix is ambiguous — the
technical report shows one OJBench column, while public leaderboards track
Python and C++ subsets separately. `baseline_all.jsonl` gives you mean pass@1
over the same rollouts CLR selects from. That is the only apples-to-apples
comparison. If you also want to check whether you reproduce ~38.6, that is a
separate experiment, and say so.

**Report K, and report variance.** `--splits 2` partitions the 16 candidates
into two disjoint groups of 8 and selects independently in each, so you get two
measurements instead of one. The report runs its CLR flow 8 times and averages;
you can't afford that, but two is much better than one.

---

## 8. Files

| File | Where | What |
| --- | --- | --- |
| `config.yaml` | both | every knob; CLI flags override |
| `common.py` | both | jsonl, code extraction, sample parsing, sandboxed runner |
| `llm.py` | GPU | async vLLM client, retries, progress |
| `setup_vast.sh` / `serve_vllm.sh` | GPU | install and serve |
| `step1_gen_candidates.py` | GPU | K rollouts; `--mode generator` for input generators |
| `step2_local_tests.py` | GPU | compile, sample tests, behavioural signature |
| `step3_extract_claims.py` | GPU | M claims per candidate |
| `step4_verify_claims.py` | GPU | adversarial self-verification → verdicts |
| `step5_select.py` | GPU | CLR aggregation + 5 ablation variants |
| `setup_wsl.sh` | WSL2 | DMOJ + OJBench + testdata |
| `step6_judge.py` | WSL2 | judge and report |

## 9. References

- VibeThinker-3B report — <https://arxiv.org/abs/2606.16140> (CLR: §3.1)
- Model — <https://huggingface.co/WeiboAI/VibeThinker-3B>
- OJBench — <https://github.com/He-Ren/OJBench>, data at
  <https://huggingface.co/datasets/He-Ren/OJBench_testdata>
