#!/usr/bin/env python3
"""Coverage for the failover harness's self-check and its pre-registered rule.

The harness cannot be exercised without a live two-replica cluster, which is exactly
why the parts that decide whether a run is worth starting need testing offline. Every
assertion here is about a run that has not happened yet: can the configuration observe
the rule it will be judged by, and does the rule say what it means.
"""
import argparse
import io
import json
import os
import re
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from benchmarks.run_failover_bench import (  # noqa: E402
    RULE,
    build_parser,
    check_kill_command,
    cmd_self_check,
    cmd_verdict,
)

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

GOOD_KILL = "kubectl delete pod kimi-k3-serving-0 -n llm-serving --wait=false"
GOOD_REJOIN = (
    "kubectl get pod kimi-k3-serving-0 -n llm-serving"
    " -o jsonpath='{.status.conditions[?(@.type==\"Ready\")].status}'"
)


def parse_measure(**overrides):
    """Build a measure-args namespace the way the CLI would, then apply overrides."""
    argv = ["measure", "--endpoint=http://gateway.invalid:4000",
            f"--kill-command={GOOD_KILL}", f"--rejoin-command={GOOD_REJOIN}"]
    args = build_parser().parse_args(argv)
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def run_self_check(**overrides):
    """Return (exit_code, combined_output) without touching a cluster."""
    args = parse_measure(**overrides)
    out, err = io.StringIO(), io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        rc = cmd_self_check(args)
    return rc, out.getvalue() + err.getvalue()


class TestSelfCheckCanObserveItsOwnRule(unittest.TestCase):
    """A run whose outcome is decided by its clock is misconfigured, not unlucky."""

    def test_default_recovery_window_outlasts_the_rejoin_limit(self):
        # The regression this file exists for. poll_rejoin is stopped the moment the
        # recovery phase ends, so a default below max_rejoin_seconds guarantees a null
        # rejoin_seconds and therefore a failed verdict, at four-node prices.
        default = build_parser().parse_args(
            ["measure", "--endpoint=http://x.invalid", f"--kill-command={GOOD_KILL}"]
        ).recovery_seconds
        self.assertGreaterEqual(
            default, RULE["max_rejoin_seconds"],
            "the default recovery window stops the rejoin poller before the rejoin"
            " limit it is judged against; the run would fail on a rule it never measured",
        )

    def test_default_configuration_passes_self_check(self):
        rc, _ = run_self_check()
        self.assertEqual(rc, 0, "the shipped defaults must not be a misconfiguration")

    def test_recovery_window_below_the_rejoin_limit_is_rejected(self):
        rc, output = run_self_check(recovery_seconds=900)
        self.assertEqual(rc, 1)
        self.assertIn("rejoin limit", output)

    def test_recovery_window_below_the_blackout_limit_is_rejected(self):
        rc, output = run_self_check(recovery_seconds=30)
        self.assertEqual(rc, 1)
        self.assertIn("blackout limit", output)

    def test_missing_rejoin_command_is_rejected(self):
        # verdict fails a null rejoin_seconds unconditionally, so a run without a rejoin
        # command has already lost before the replica is killed.
        rc, output = run_self_check(rejoin_command="")
        self.assertEqual(rc, 1)
        self.assertIn("rejoin-command", output)

    def test_short_baseline_is_rejected(self):
        rc, output = run_self_check(baseline_seconds=5)
        self.assertEqual(rc, 1)
        self.assertIn("baseline-seconds", output)


class TestKillCommandIsBounded(unittest.TestCase):
    """The harness deletes things. What it may delete is a fixed list."""

    def test_a_pod_delete_is_allowed(self):
        self.assertEqual(check_kill_command(GOOD_KILL), [])

    def test_destructive_commands_are_refused(self):
        for command in (
            "kubectl delete namespace llm-serving",
            "kubectl delete statefulset kimi-k3-serving -n llm-serving",
            "kubectl delete pod --all -n llm-serving",
            "terraform destroy -auto-approve",
            "bash scripts/06_destroy_all.sh",
            "kubectl delete pvc kimi-k3-weights-rox -n llm-serving",
        ):
            with self.subTest(command=command):
                self.assertNotEqual(
                    check_kill_command(command), [],
                    f"{command!r} would destroy more than one replica's leader",
                )

    def test_self_check_refuses_an_unsafe_kill(self):
        rc, output = run_self_check(kill_command="kubectl delete namespace llm-serving")
        self.assertEqual(rc, 1)
        self.assertIn("unsafe --kill-command", output)


def verdict_of(doc):
    """Run the offline verdict over a document and return its parsed output."""
    with tempfile.TemporaryDirectory() as tmp:
        src = os.path.join(tmp, "in.json")
        dst = os.path.join(tmp, "out.json")
        with open(src, "w", encoding="utf-8") as fh:
            json.dump(doc, fh)
        args = argparse.Namespace(input=src, output=dst)
        buf = io.StringIO()
        with redirect_stdout(buf), redirect_stderr(buf):
            rc = cmd_verdict(args)
        with open(dst, encoding="utf-8") as fh:
            return rc, json.load(fh)


def passing_doc(**summary_overrides):
    summary = {
        "blackout_seconds": 12.0,
        "baseline_error_rate": 0.0,
        "residual_error_rate": 0.0,
        "residual_requests": 400,
        "baseline_p99_s": 1.0,
        "recovery_p99_s": 1.5,
        "failures_after_kill": 20,
        "seconds_to_first_success_after_kill": 12.0,
    }
    summary.update(summary_overrides)
    return {"summary": summary, "rejoin_seconds": 1020.0, "kill": {"kill_rc": 0}}


class TestVerdictSaysWhatItMeans(unittest.TestCase):
    def test_a_clean_failover_passes(self):
        rc, doc = verdict_of(passing_doc())
        self.assertEqual(rc, 0, doc)
        self.assertTrue(all(c["pass"] for c in doc["checks"]), doc["checks"])

    def test_an_unmeasured_rejoin_fails_and_says_so(self):
        # Documents the behaviour that motivated the default change: null is a failure,
        # and the detail string has to distinguish it from a replica that stayed down.
        rc, doc = verdict_of({**passing_doc(), "rejoin_seconds": None})
        self.assertEqual(rc, 2, "a failed rule must exit 2, as sweep_decision.py does")
        self.assertEqual(doc["verdict"], "FAIL")
        rejoin = next(c for c in doc["checks"] if c["check"] == "rejoin")
        self.assertFalse(rejoin["pass"])
        self.assertIn("not measured", rejoin["detail"])

    def test_a_long_blackout_fails(self):
        rc, doc = verdict_of(passing_doc(
            blackout_seconds=RULE["max_blackout_seconds"] + 1))
        self.assertEqual(rc, 2)
        self.assertFalse(next(c for c in doc["checks"] if c["check"] == "blackout")["pass"])

    def test_a_sick_baseline_invalidates_the_run(self):
        rc, doc = verdict_of(passing_doc(
            baseline_error_rate=RULE["max_baseline_error_rate"] + 0.01))
        self.assertEqual(rc, 2)
        self.assertFalse(
            next(c for c in doc["checks"] if c["check"] == "baseline_health")["pass"])

    def test_the_rule_bounds_are_mutually_satisfiable(self):
        """The first draft of this rule could not be passed by any run.

        Whole-run error rate is duration-dependent, so a run that exactly met the
        blackout limit automatically failed the error limit. Residual error rate fixed
        that; this asserts the arithmetic stays consistent if the constants move.
        """
        rc, _ = verdict_of(passing_doc(
            blackout_seconds=RULE["max_blackout_seconds"],
            residual_error_rate=RULE["max_residual_error_rate"],
            baseline_error_rate=RULE["max_baseline_error_rate"],
            recovery_p99_s=RULE["max_recovery_p99_ratio"],
            baseline_p99_s=1.0,
        ))
        self.assertEqual(rc, 0, "a run sitting exactly on every limit must be a pass")


class TestTheWiredCallerMatchesTheHarness(unittest.TestCase):
    """05_run_benchmarks.sh is the only caller; drift between them is silent."""

    SCRIPT = os.path.join(REPO_ROOT, "scripts", "05_run_benchmarks.sh")

    def setUp(self):
        with open(self.SCRIPT, encoding="utf-8") as fh:
            self.src = fh.read()

    def test_self_check_validates_the_same_args_as_the_run(self):
        # Both invocations must expand FO_ARGS. Validating a subset is how a run passes
        # its pre-flight and then fails on the field the pre-flight never saw.
        invocations = re.findall(r"run_failover_bench\.py\"? ([^\n]*)", self.src)
        checked = [i for i in invocations if "--self-check" in i]
        self.assertTrue(checked, "no self-check invocation found in the wired caller")
        for inv in checked:
            self.assertIn(
                "FO_ARGS", inv,
                "the self-check must validate the argument array the measurement uses",
            )

    def test_the_caller_supplies_a_rejoin_command(self):
        self.assertIn("--rejoin-command=", self.src)

    def test_rejoin_expect_matches_what_kubectl_returns(self):
        # kubectl renders a Ready condition status as "True"; the harness compares with
        # ==, so a lowercase expectation would never match and rejoin would read as null.
        self.assertIn("--rejoin-expect=True", self.src)


if __name__ == "__main__":
    unittest.main()
