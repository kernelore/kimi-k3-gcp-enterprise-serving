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

## Suite ordering

The provenance gate in `generate_comparison.py` requires the suites to have been run in the
order `standard → massive → soak → saturation → prefill`, with each suite's interval
starting no earlier than the previous suite's end. The integrity audit independently checks
the same thing using `metadata.run_timestamp` plus each suite's measured duration. A partial
result set is a hard error in both — publish all five suites for an engine, or none.
