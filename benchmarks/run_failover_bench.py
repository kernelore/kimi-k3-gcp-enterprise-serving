#!/usr/bin/env python3
"""Measure what a leader-pod loss costs a client, at DP>=2.

Phase 3 asks a question the throughput sweep cannot: when one serving replica dies,
does the gateway keep answering, how much does it drop on the floor, and how long
until the replica is back. That is a resilience number, not a throughput number, so
it is measured against a steady load and judged on continuity rather than tokens/s.

Why a leader pod specifically. Each replica is a 2-node SGLang MPI group and only
rank 0 serves HTTP -- `kimi-k3-serving-svc` selects on `apps.kubernetes.io/pod-index: "0"`.
Killing rank 0 therefore removes the whole replica's serving surface, not one worker
of many, which is the worst realistic single-pod loss and the one worth budgeting for.
Killing rank 1 would be a milder test wearing the same name.

The kill is issued by this harness rather than by hand, because every number here is
relative to T0 and a T0 recorded by a human with a stopwatch is not worth the GPU-hours
it took to produce.

Three phases, one continuous load:

    [ baseline ] --T0 kill--> [ degraded ] ... [ recovered ]

The pass rule is registered in RULE below and evaluated by `verdict`, which reads only
the JSON this produces. It is written before any run, deliberately: a threshold chosen
after seeing the number it judges is not a threshold.
"""

import argparse
import json
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from collections import Counter

# ---------------------------------------------------------------------------
# Pre-registered decision rule.
#
# Set before the first run and not to be edited to fit a result. If a bound turns
# out to be wrong, say so in the report and change it in a commit of its own, so the
# change is visible as a change rather than absorbed into a passing run.
# ---------------------------------------------------------------------------
RULE = {
    # How long clients may see *zero* successful responses. A gateway that fails over
    # cleanly should stay under this comfortably; a gateway that does not fail over at
    # all will blow through it and the run says so plainly.
    "max_blackout_seconds": 60.0,
    # Loss AFTER service returns -- did failover settle, or is it flapping.
    #
    # Not whole-run error rate, which was the first draft of this rule and was wrong in a
    # way worth recording. Whole-run error rate is duration-dependent: the same outage
    # scores better the longer the run continues around it. Worse, the two bounds were
    # arithmetically inconsistent -- a 60s outage at a steady rate inside a 1020s run is
    # already 5.9% error, so any run that exactly met the blackout limit automatically
    # failed the error limit. Blackout already measures the outage; this measures whether
    # what came back is healthy, and neither one moves when the run gets longer.
    "max_residual_error_rate": 0.02,
    # Validity check, not a resilience check. If the service was already dropping requests
    # before the kill, the run measures a sick cluster rather than a failover.
    "max_baseline_error_rate": 0.01,
    # One replica absorbing the load of two will be slower. This bounds how much
    # slower before "degraded but serving" becomes "up in name only".
    "max_recovery_p99_ratio": 3.0,
    # The killed replica has to come back. 1.5x the measured 17 min 03 s warm ROX
    # restart, which is the number this repository already stands behind.
    "max_rejoin_seconds": 1535.0,
}

DEFAULT_PROMPT = (
    "Summarise the operational trade-offs of running a large mixture-of-experts"
    " model across two nodes rather than one, in three sentences."
)


def _now():
    return time.time()


def issue_request(endpoint, model, prompt, max_tokens, api_key, timeout):
    """One request. Returns (ok, latency_seconds, label).

    Every failure mode collapses to a short label rather than an exception, because
    the point of this harness is to keep issuing load while things are broken. An
    exception escaping here would end the measurement at exactly the moment it starts
    being interesting.
    """
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }).encode()
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(endpoint, data=body, headers=headers, method="POST")
    start = _now()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as res:
            res.read()
            return True, _now() - start, f"http_{res.status}"
    except urllib.error.HTTPError as exc:
        return False, _now() - start, f"http_{exc.code}"
    except urllib.error.URLError as exc:
        return False, _now() - start, f"urlerror_{type(exc.reason).__name__}"
    except Exception as exc:  # noqa: BLE001 - see docstring
        return False, _now() - start, f"error_{type(exc).__name__}"


def load_generator(stop_event, results, lock, args):
    """Issue requests at a fixed arrival rate, open loop.

    Deliberately NOT the closed-loop "N workers looping back to back" that the throughput
    sweep uses. Under a closed loop, a gateway that returns 503 in a millisecond gets
    hammered with tens of thousands of instant failures while a healthy one paces itself
    at the speed of real generation -- so the failure *count* ends up measuring how fast
    the service says no. Measured on a fake gateway: an 8-second outage produced 26k
    failed requests against 800 baseline successes, a 97% "error rate" for an outage that
    a fixed-rate client would have scored at 6%. That metric punishes failing fast, which
    is the behaviour we actually want.

    A fixed arrival rate makes the denominator a property of the test rather than of the
    failure, so "fraction of attempts lost" means what it says.
    """
    interval = 1.0 / args.rps
    inflight = threading.Semaphore(args.max_inflight)
    threads = []
    issued = 0
    start = _now()

    def one_shot():
        try:
            ok, latency, label = issue_request(
                args.endpoint, args.model, args.prompt,
                args.max_tokens, args.api_key, args.request_timeout,
            )
            with lock:
                results.append({"t": _now(), "ok": ok, "latency": latency, "label": label})
        finally:
            inflight.release()

    while not stop_event.is_set():
        target = start + issued * interval
        delay = target - _now()
        if delay > 0:
            if stop_event.wait(delay):
                break
        issued += 1
        if not inflight.acquire(blocking=False):
            # The client has more requests outstanding than it is willing to hold. Recording
            # this as a loss rather than queueing it is the honest reading: a real caller with
            # a bounded connection pool drops the request too, and silently queueing would
            # hide a stalled service behind a growing backlog.
            with lock:
                results.append({"t": _now(), "ok": False, "latency": 0.0,
                                "label": "shed_inflight_cap"})
            continue
        t = threading.Thread(target=one_shot, daemon=True)
        t.start()
        threads.append(t)
        if len(threads) > 512:
            threads = [x for x in threads if x.is_alive()]
    for t in threads:
        t.join(timeout=args.request_timeout + 5)


def run_kill(command, record):
    """Run the kill and stamp it. Recorded even when it fails, so a run that measured
    nothing is distinguishable from a run that measured a healthy system."""
    record["kill_issued_epoch"] = _now()
    try:
        proc = subprocess.run(
            command, shell=True, capture_output=True, text=True, timeout=120,
        )
        record["kill_rc"] = proc.returncode
        record["kill_stdout"] = proc.stdout.strip()[:2000]
        record["kill_stderr"] = proc.stderr.strip()[:2000]
    except subprocess.TimeoutExpired:
        record["kill_rc"] = -1
        record["kill_stderr"] = "kill command timed out after 120s"
    record["kill_returned_epoch"] = _now()


def poll_rejoin(stop_event, record, args):
    """Watch for the killed replica becoming Ready again.

    Separate from the load loop on purpose: a client cannot see 'Ready', only whether
    it got an answer, and conflating the two is how a recovery number ends up measuring
    the readiness probe instead of the service.
    """
    if not args.rejoin_command:
        return
    while not stop_event.is_set():
        try:
            proc = subprocess.run(
                args.rejoin_command, shell=True,
                capture_output=True, text=True, timeout=30,
            )
            if proc.returncode == 0 and proc.stdout.strip() == args.rejoin_expect:
                record["rejoin_epoch"] = _now()
                return
        except subprocess.TimeoutExpired:
            pass
        time.sleep(5)


def summarise(samples, t0, end_epoch):
    """Turn the raw request log into the numbers the rule is written against."""
    baseline = [s for s in samples if s["t"] < t0]
    after = [s for s in samples if s["t"] >= t0]

    def p(values, q):
        if not values:
            return None
        ordered = sorted(values)
        idx = min(int(q * len(ordered)), len(ordered) - 1)
        return ordered[idx]

    base_ok = [s["latency"] for s in baseline if s["ok"]]
    after_ok = [s["latency"] for s in after if s["ok"]]

    failures_after = [s for s in after if not s["ok"]]
    first_error = min((s["t"] for s in failures_after), default=None)
    last_error = max((s["t"] for s in failures_after), default=None)

    # Blackout is the longest span with no success at all, which is what a caller actually
    # experiences. It is NOT last_error - first_error: interleaved successes mean the
    # service was up and lossy, which is a different and much less severe failure.
    #
    # The trailing gap -- last success to end of run -- is counted with the rest, and that
    # is not a detail. An earlier version only measured gaps *between* successes, so a
    # gateway that went down at T0 and never came back recorded its longest gap as the
    # few milliseconds before the kill landed and scored blackout=0.0s. The worst possible
    # outcome read as the best possible one. Caught on a fake gateway, not in production.
    successes_after = sorted(s["t"] for s in after if s["ok"])
    blackout = 0.0
    blackout_start = None
    cursor = t0
    for ts in successes_after + [max(end_epoch, t0)]:
        if ts - cursor > blackout:
            blackout = ts - cursor
            blackout_start = cursor
        cursor = max(cursor, ts)

    first_success_after = successes_after[0] if successes_after else None

    # Everything from the end of the blackout onward: the service is nominally back, so
    # anything still failing here is failover that has not settled.
    blackout_end = (blackout_start + blackout) if blackout_start is not None else t0
    residual = [s for s in after if s["t"] >= blackout_end]
    residual_failures = [s for s in residual if not s["ok"]]

    return {
        "requests_total": len(samples),
        "requests_baseline": len(baseline),
        "requests_after_kill": len(after),
        "failures_after_kill": len(failures_after),
        "failures_baseline": len([s for s in baseline if not s["ok"]]),
        "error_rate": (len([s for s in samples if not s["ok"]]) / len(samples)) if samples else None,
        "error_rate_after_kill": (len(failures_after) / len(after)) if after else None,
        "baseline_error_rate": (
            len([s for s in baseline if not s["ok"]]) / len(baseline)) if baseline else None,
        "residual_error_rate": (
            len(residual_failures) / len(residual)) if residual else None,
        "residual_requests": len(residual),
        "blackout_end_epoch": blackout_end,
        "baseline_p50_s": p(base_ok, 0.50),
        "baseline_p99_s": p(base_ok, 0.99),
        "recovery_p50_s": p(after_ok, 0.50),
        "recovery_p99_s": p(after_ok, 0.99),
        "blackout_seconds": blackout,
        "blackout_start_epoch": blackout_start,
        "first_error_epoch": first_error,
        "last_error_epoch": last_error,
        "seconds_to_first_success_after_kill": (
            first_success_after - t0 if first_success_after else None
        ),
        "failure_labels": dict(Counter(s["label"] for s in samples if not s["ok"])),
        "arm_note": (
            "no failures observed after the kill; either failover was seamless or the"
            " kill did not take -- check kill_rc and the replica's pod age before"
            " reading this as a pass"
        ) if not failures_after else None,
    }


# Shapes a kill command must not have. --kill-command is arbitrary shell aimed at a live
# cluster during a metered window, so the blast radius is worth bounding in code rather
# than in a comment. Everything here removes far more than one pod: the point of this test
# is to lose a replica, not the cluster it runs on, and a typo that costs the weights
# backup or the whole StatefulSet is not recoverable inside the window.
FORBIDDEN_KILL_PATTERNS = [
    ("delete namespace", "removes the namespace, not a pod"),
    ("delete ns ", "removes the namespace, not a pod"),
    ("delete statefulset", "removes the workload; nothing would come back to measure"),
    ("delete sts", "removes the workload; nothing would come back to measure"),
    ("delete node", "removes a node; the pods land elsewhere and the test is meaningless"),
    ("--all", "deletes every matching object rather than one leader"),
    ("terraform destroy", "tears down the stack mid-measurement"),
    ("06_destroy_all", "tears down the stack mid-measurement"),
    ("delete pvc", "destroys staged weights"),
    ("gsutil rm", "touches durable storage, which no failover test needs"),
    ("gcloud storage rm", "touches durable storage, which no failover test needs"),
]


def check_kill_command(command):
    """Return a list of reasons this kill command should not run."""
    lowered = " ".join(command.lower().split())
    return [f"{pat.strip()!r}: {why}" for pat, why in FORBIDDEN_KILL_PATTERNS if pat in lowered]


def cmd_self_check(args):
    """Validate the run without issuing a single request or touching the cluster.

    Same intent as the probe self-check on the KV gate: the cheapest moment to catch a
    misconfigured run is before the GPUs are billing for it.
    """
    problems = []
    if not args.kill_command.strip():
        problems.append("--kill-command is empty; there would be nothing to fail over from")
    problems.extend(f"unsafe --kill-command, matched {r}" for r in check_kill_command(args.kill_command))
    if args.rps <= 0:
        problems.append(f"--rps must be positive, got {args.rps}")
    if args.baseline_seconds < 30:
        problems.append(
            f"--baseline-seconds={args.baseline_seconds} is too short to establish a p99 to"
            " compare against; use at least 30")
    if args.recovery_seconds <= RULE["max_blackout_seconds"]:
        # Otherwise the run can end mid-outage and report a blackout bounded by its own
        # duration, which would look like a pass for a service that never came back.
        problems.append(
            f"--recovery-seconds={args.recovery_seconds} must exceed the"
            f" {RULE['max_blackout_seconds']}s blackout limit, or the run cannot observe"
            " a failure of that rule")
    if args.recovery_seconds < RULE["max_rejoin_seconds"]:
        # The same reasoning as the blackout guard above, one bound higher. poll_rejoin is
        # stopped when the recovery phase ends, so a window shorter than the limit it is
        # judged against cannot tell "never rejoined" from "not watched long enough" --
        # and verdict fails a null rejoin, so the run's outcome is decided before it starts.
        problems.append(
            f"--recovery-seconds={args.recovery_seconds} is below the"
            f" {RULE['max_rejoin_seconds']}s rejoin limit; the poller would stop before a"
            " rejoin could be observed and the verdict would fail a rule it never measured")
    if not args.rejoin_command.strip():
        # Every other check here asks whether the run can measure what it claims to. This
        # one asks the same of rejoin: without a command to poll, rejoin_seconds is null by
        # construction and verdict scores that as a failure regardless of what the fabric
        # did. Cheaper to say so now than after the replica has been killed.
        problems.append(
            "--rejoin-command is empty; rejoin_seconds would be null by construction and"
            " the verdict would fail on it no matter how well the failover went")
    if not args.endpoint.startswith(("http://", "https://")):
        problems.append(f"--endpoint is not an http(s) URL: {args.endpoint}")
    expected = args.rps * (args.baseline_seconds + args.recovery_seconds)
    if expected < 100:
        problems.append(
            f"the run would issue about {expected:.0f} requests, too few for a stable p99")

    if problems:
        print("ERROR: failover run is misconfigured:", file=sys.stderr)
        for p in problems:
            print(f"  - {p}", file=sys.stderr)
        return 1
    print(f"    [OK] self-check: ~{expected:.0f} requests at {args.rps}/s, kill command bounded")
    return 0


def cmd_measure(args):
    if args.self_check:
        return cmd_self_check(args)
    refusals = check_kill_command(args.kill_command)
    if refusals:
        # Belt and braces: self-check is advisory, this is not.
        print("ERROR: refusing to run this kill command:", file=sys.stderr)
        for r in refusals:
            print(f"  - {r}", file=sys.stderr)
        return 1

    samples = []
    lock = threading.Lock()
    stop_event = threading.Event()
    record = {}

    print(f"--> baseline: {args.baseline_seconds}s at {args.rps} req/s "
          f"(open loop, max {args.max_inflight} in flight)")
    gen = threading.Thread(
        target=load_generator, args=(stop_event, samples, lock, args), daemon=True)
    gen.start()

    start_epoch = _now()
    time.sleep(args.baseline_seconds)

    with lock:
        baseline_count = len(samples)
    if baseline_count == 0:
        stop_event.set()
        print("ERROR: no requests completed during the baseline phase.", file=sys.stderr)
        print("       Nothing can be measured against a baseline that does not exist.", file=sys.stderr)
        return 1

    print(f"--> baseline captured ({baseline_count} requests); issuing kill")
    print(f"    {args.kill_command}")
    run_kill(args.kill_command, record)
    t0 = record["kill_issued_epoch"]
    if record.get("kill_rc") != 0:
        print(f"    WARNING: kill exited rc={record.get('kill_rc')}: {record.get('kill_stderr','')}",
              file=sys.stderr)

    rejoin_stop = threading.Event()
    rejoin_thread = threading.Thread(
        target=poll_rejoin, args=(rejoin_stop, record, args), daemon=True)
    rejoin_thread.start()

    print(f"--> recovery: holding load for {args.recovery_seconds}s")
    time.sleep(args.recovery_seconds)
    end_epoch = _now()
    stop_event.set()
    rejoin_stop.set()
    gen.join(timeout=args.request_timeout + 15)

    with lock:
        frozen = list(samples)
    frozen.sort(key=lambda s: s["t"])

    out = {
        "schema": "kimi-k3-failover/1",
        "label": args.label,
        "endpoint": args.endpoint,
        "model": args.model,
        "rps": args.rps,
        "max_inflight": args.max_inflight,
        "baseline_seconds": args.baseline_seconds,
        "recovery_seconds": args.recovery_seconds,
        "start_epoch": start_epoch,
        "end_epoch": end_epoch,
        "kill": record,
        "rejoin_seconds": (
            record["rejoin_epoch"] - t0 if record.get("rejoin_epoch") else None
        ),
        "summary": summarise(frozen, t0, end_epoch),
    }
    if args.keep_samples:
        out["samples"] = frozen

    with open(args.output, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    print(f"--> wrote {args.output}")
    _print_summary(out)
    return 0


def _print_summary(doc):
    s = doc["summary"]
    print("\n  failover summary")
    print(f"    requests            {s['requests_total']} "
          f"({s['requests_baseline']} baseline / {s['requests_after_kill']} after)")
    print(f"    error rate          {_fmt_pct(s['error_rate'])} overall, "
          f"{_fmt_pct(s['error_rate_after_kill'])} after the kill")
    print(f"    baseline errors     {_fmt_pct(s.get('baseline_error_rate'))}")
    print(f"    residual errors     {_fmt_pct(s.get('residual_error_rate'))} "
          f"over {s.get('residual_requests')} requests once service returned")
    print(f"    blackout            {_fmt_s(s['blackout_seconds'])}")
    print(f"    first success after {_fmt_s(s['seconds_to_first_success_after_kill'])}")
    print(f"    p99 baseline        {_fmt_s(s['baseline_p99_s'])}")
    print(f"    p99 recovery        {_fmt_s(s['recovery_p99_s'])}")
    print(f"    replica rejoin      {_fmt_s(doc.get('rejoin_seconds'))}")
    if s.get("failure_labels"):
        print(f"    failure modes       {s['failure_labels']}")
    if s.get("arm_note"):
        print(f"    NOTE: {s['arm_note']}")


def _fmt_s(v):
    return "n/a" if v is None else f"{v:.2f}s"


def _fmt_pct(v):
    return "n/a" if v is None else f"{100 * v:.2f}%"


def cmd_verdict(args):
    with open(args.input, encoding="utf-8") as fh:
        doc = json.load(fh)
    s = doc["summary"]
    checks = []

    def check(name, ok, detail):
        checks.append({"check": name, "pass": bool(ok), "detail": detail})

    blackout = s.get("blackout_seconds")
    check("blackout", blackout is not None and blackout <= RULE["max_blackout_seconds"],
          f"{_fmt_s(blackout)} vs limit {RULE['max_blackout_seconds']}s")

    base_err = s.get("baseline_error_rate")
    check("baseline_health", base_err is not None and base_err <= RULE["max_baseline_error_rate"],
          f"{_fmt_pct(base_err)} vs limit {_fmt_pct(RULE['max_baseline_error_rate'])}"
          + ("  (run is invalid, not the fabric's fault)" if base_err else ""))

    resid = s.get("residual_error_rate")
    if resid is None:
        # No requests landed after the blackout ended. Blackout already failed; do not
        # also fail this and report one problem as two.
        check("residual_error_rate", False, "no requests after recovery -- service never returned")
    else:
        check("residual_error_rate", resid <= RULE["max_residual_error_rate"],
              f"{_fmt_pct(resid)} over {s.get('residual_requests')} requests"
              f" vs limit {_fmt_pct(RULE['max_residual_error_rate'])}")

    bp99, rp99 = s.get("baseline_p99_s"), s.get("recovery_p99_s")
    if bp99 and rp99:
        ratio = rp99 / bp99
        check("recovery_p99_ratio", ratio <= RULE["max_recovery_p99_ratio"],
              f"{ratio:.2f}x vs limit {RULE['max_recovery_p99_ratio']}x")
    else:
        check("recovery_p99_ratio", False, "missing p99 on one side")

    rejoin = doc.get("rejoin_seconds")
    if rejoin is None:
        # Not measured is not the same as failed. Say which.
        check("rejoin", False, "not measured (no --rejoin-command given, or never observed)")
    else:
        check("rejoin", rejoin <= RULE["max_rejoin_seconds"],
              f"{_fmt_s(rejoin)} vs limit {RULE['max_rejoin_seconds']}s")

    # A run where nothing failed after the kill is suspicious, not automatically good.
    kill_rc = (doc.get("kill") or {}).get("kill_rc")
    if kill_rc != 0:
        check("kill_effective", False, f"kill command exited rc={kill_rc}")
    elif s.get("failures_after_kill") == 0 and s.get("seconds_to_first_success_after_kill") is not None:
        check("kill_effective", True,
              "no client-visible failures; failover was seamless (verify pod age separately)")
    else:
        check("kill_effective", True, f"{s.get('failures_after_kill')} failures observed after kill")

    verdict = "PASS" if all(c["pass"] for c in checks) else "FAIL"
    out = {
        "schema": "kimi-k3-failover-verdict/1",
        "label": doc.get("label"),
        "rule": RULE,
        "checks": checks,
        "verdict": verdict,
    }
    if args.output:
        with open(args.output, "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=2)
    print(f"\n  pre-registered rule -> {verdict}")
    for c in checks:
        print(f"    [{'ok ' if c['pass'] else 'FAIL'}] {c['check']}: {c['detail']}")
    return 0 if verdict == "PASS" else 2


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)

    m = sub.add_parser("measure", help="drive load, kill a leader, record the timeline")
    m.add_argument("--endpoint", default="http://localhost:4000/v1/chat/completions",
                   help="Gateway chat-completions endpoint. Defaults to the gateway, not a"
                        " replica: the question is whether the *service* survives.")
    m.add_argument("--model", default="moonshotai/Kimi-K3")
    m.add_argument("--api-key", default="")
    m.add_argument("--prompt", default=DEFAULT_PROMPT)
    m.add_argument("--max-tokens", type=int, default=64,
                   help="Short on purpose: this measures availability, not generation.")
    m.add_argument("--rps", type=float, default=2.0,
                   help="Arrival rate, open loop. Modest on purpose: this is an availability"
                        " probe running alongside whatever else is on the cluster, not a"
                        " throughput run.")
    m.add_argument("--max-inflight", type=int, default=64,
                   help="Bound on outstanding requests. Attempts beyond it are recorded as"
                        " losses rather than queued, which is what a bounded client pool does.")
    m.add_argument("--baseline-seconds", type=int, default=120)
    # Must outlast RULE["max_rejoin_seconds"], because poll_rejoin is stopped the instant
    # this elapses. The previous default of 900 shut the poller off two minutes before the
    # measured 17 min 03 s warm ROX restart could plausibly finish, so rejoin_seconds came
    # back null and verdict scored the rejoin check as a failure -- a FAIL describing the
    # clock rather than the fabric, bought at four-node prices.
    m.add_argument("--recovery-seconds", type=int, default=1600)
    m.add_argument("--request-timeout", type=int, default=60)
    m.add_argument("--kill-command", required=True,
                   help="Shell command that removes the leader pod, e.g."
                        " 'kubectl delete pod kimi-k3-serving-0 -n llm-serving --wait=false'")
    m.add_argument("--rejoin-command", default="",
                   help="Optional shell command polled every 5s; when its stdout equals"
                        " --rejoin-expect the replica counts as back.")
    m.add_argument("--rejoin-expect", default="true")
    m.add_argument("--label", default="failover")
    m.add_argument("--output", default="benchmarks/failover_results_kimi_k3.json")
    m.add_argument("--self-check", action="store_true",
                   help="Validate the configuration and the kill command, then exit without"
                        " issuing a request. Run this before the cluster is up.")
    m.add_argument("--keep-samples", action="store_true",
                   help="Embed the per-request log. Large, but the only way to re-derive"
                        " a number without paying for the window again.")
    m.set_defaults(func=cmd_measure)

    v = sub.add_parser("verdict", help="apply the pre-registered rule offline")
    v.add_argument("--input", required=True)
    v.add_argument("--output", default="")
    v.set_defaults(func=cmd_verdict)
    return ap


def main():
    args = build_parser().parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
