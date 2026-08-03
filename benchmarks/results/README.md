# Raw benchmark results

One JSON file per suite per engine, exactly as the harness in `benchmarks/` wrote it.
`benchmarks/generate_comparison.py` reads this tree to build the comparison block in the
top-level `README.md`, and `tests/adv_audit_benchmark_integrity.py` audits it in CI.

Files are committed verbatim. They are never hand-edited — a number in this tree is a
number the harness measured, which is the whole point of auditing it.

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
