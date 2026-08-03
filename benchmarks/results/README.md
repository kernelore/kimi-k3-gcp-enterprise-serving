# Raw benchmark results

One JSON file per suite per engine, exactly as the harness in `benchmarks/` wrote it.
`benchmarks/generate_comparison.py` reads this tree to build the comparison block in the
top-level `README.md`, and `tests/adv_audit_benchmark_integrity.py` audits it in CI.

Files are committed verbatim. They are never hand-edited — a number in this tree is a
number the harness measured, which is the whole point of auditing it.

One file is not like that and says so in its own `provenance` field:
`sglang/dspark_prompt_sensitivity_c16.json` was assembled by hand from two sources, and
no harness run produced it. It is the exception the rule is stated against, not a
counter-example to it. See "The hand-assembled file" below.

## Which engine a file belongs to

`metadata.engine` is authoritative. Read that, not the top-level `engine` key.

The top-level `engine` key simply echoes the harness's `--engine` argument. Every harness
defaults that argument to `trtllm`, and the in-cluster Job template did not pass it until
`scripts/05_run_benchmarks.sh` was fixed to forward the selected engine. The
`standard`, `massive` and `soak` files under `sglang/` were produced before that fix and
therefore carry `"engine": "trtllm"` at the top level while their `metadata.engine` — which
is derived from the deployed image and is what every consumer reads — correctly says
`sglang`. The measurements in those files are unaffected: the argument is a label, and
nothing about how a request was issued or timed depends on it.

They have been left as measured rather than corrected in place. Runs made after the fix
carry the right value in both places.

## Which image produced these files

Every file in `sglang/` was measured against base digest
`lmsysorg/sglang:kimi-k3@sha256:81a9c006…`. `docker/Dockerfile.sglang` has since moved to
`sha256:6d9594a4…`, which is the same v0.5.16 base plus a source overlay from
`sgl-project/sglang@c6ad1f26` — about 225 commits of newer Python over the `srt` and
`kernels` trees, with the compiled kernel wheel left at the older build.

Nothing in this tree was re-measured for that move, and no gate catches it: the recorded
`metadata.engine_version` is derived from the image *tag* (`kimi-k3`), which did not change,
so the provenance checks pass either way. Treat a comparison between these numbers and
anything measured on `6d9594a4…` as cross-image, not like-for-like.

## Saturation prompt calibration

`saturation_results.json` records the calibration constant its run used in
`grid.BASE_TOKENS_APPROX`. The committed sweep used `1024`, the value the synthetic
passage was written to approximate. Kimi K3's tokenizer in fact renders that passage as
about 889 tokens, so every cell built a prompt 11–13% shorter than its grid label — the
`prompt_tokens_observed` field on each cell records what was actually sent, and the
throughput figures are computed against those counts, not the label.

`benchmarks/run_saturation_sweep_kimi_k3.py` has since been recalibrated to `889`, so a
re-run will land the 8k and 32k targets within about 2% and will not reproduce these
prompt lengths. That is intended: the constant is a property of the tokenizer, not of the
measurement. Compare runs using `prompt_tokens_observed` and the recorded
`grid.BASE_TOKENS_APPROX`, never the grid label alone.

## Suite ordering

The provenance gate in `generate_comparison.py` requires the suites to have been run in the
order `standard → massive → soak → saturation → prefill`, with each suite's interval
starting no earlier than the previous suite's end. The integrity audit independently checks
the same thing using `metadata.run_timestamp` plus each suite's measured duration. A partial
result set is a hard error in both — publish all five suites for an engine, or none.

Those five are the whole of what is gated. Three later harnesses write into this tree and
are **not** members of the set:

| `scripts/05_run_benchmarks.sh --mode` | Harness | Writes into this tree | Gated |
|---|---|---|---|
| `realistic` | `run_realistic_sweep_kimi_k3.py` | `<engine>/realistic_results.json` | no |
| `prefix-reuse` | `run_prefix_reuse_bench.py` | `<engine>/prefix_reuse_results.json` | no |
| `kv-accuracy` | `kv_accuracy_gate.py` | `<engine>/kv_accuracy_<label>.json` | no |

None of the three is a member of `--mode all` either, so a routine full run does not
produce them.

They are excluded deliberately, not by oversight. The gate exists to stop a partial
publication of the comparison block in the top-level `README.md`, and none of these three
feeds that block — they answer tuning questions (does a variant help, does prefix reuse pay
for a third cache tier, does fp8 KV move the logits) rather than producing headline
figures. The cost of that exclusion is real and worth stating plainly: nothing checks that
a `realistic_results.json` in this tree was measured after the baseline it is being diffed
against. `benchmarks/sweep_decision.py` compensates for the one comparison that matters by
refusing any baseline/candidate pair whose two files came from different result shapes, but
it cannot check ordering, and neither can anything else here.

## The hand-assembled file

`sglang/dspark_prompt_sensitivity_c16.json` is the one file in this tree that no harness
produced. Its two arms come from two different sources measured at different times — the
repeated-passage acceptance figure was read off engine decode-batch log lines, the
non-repetitive one from a `spec_verify_calls_total` counter delta — and neither the
provenance gate nor the integrity audit looks at it, because it carries no `metadata` or
`grid` block for them to check. Its `provenance` field says all of this in the file itself,
so the caveat travels with the data rather than living only here.

It is kept because the finding is worth having: accepted tokens per verify step falls from
~6.4 to ~2.2 when prompts stop being a repeated passage, which bounds how much of README
Table 4's speculative speedup survives realistic traffic. It is a recorded observation, and
the right weight to give it is that of a careful note rather than an audited measurement.

`benchmarks/run_realistic_sweep_kimi_k3.py` was written to replace it with a file that has
both arms from one run, a `metadata` block, and an `arm_comparison` computed by the harness
instead of by hand. When that run happens, this file should be deleted rather than kept
alongside it.

## Where c=16 comes from

`benchmarks/sweep_decision.py` anchors every throughput rule on concurrency 16, and
`scripts/07_run_tuning_sweep.sh` refuses to start a sweep whose levels do not include it.
That is not an arbitrary pick. Every A/B already in this tree that was run to compare two
serving configurations — `saturation_c16_direct.json`,
`saturation_dspark_direct.json`, `saturation_tp8pp2_direct.json` — carries
`grid.sweep_levels: [8, 16, 32]` and `grid.MAX_INFLIGHT_PROMPT_TOKENS: 265000`, so c=16 is
the band with the most comparable history, and c=8/c=32 are the two neighbours those same
runs already cover, which is why they serve as the guard rails.

Note that the `saturation` suite's own default levels are `1,8,32,128` and contain no c=16
cell. A sweep left on that default cannot produce the number the decision rule reads, which
is why the driver checks the levels before spending any GPU time rather than discovering it
afterwards.
