#!/usr/bin/env bash
# standing-verdict.sh — how many REAL cog-review verdicts already judged this head?
#
# Ledger L24. `cog-review` is a required check, so any infra failure that
# publishes a blocking check-run can DEMOTE a verdict that already judged the
# same commit. On PR #605 the reviewer approved 022c4d9, a later run on the
# SAME SHA flaked three times, published verdict:error, and the failing
# check-run replaced the passing one — wedging a merge-ready PR on reviewer
# noise, and making a re-run strictly riskier than doing nothing.
#
# There is more than one infra exit (the reviewer's own `error` verdict, and
# the sandbox canary's fail-closed check). Guarding only one of them is the
# same class of bug, so the rule lives here and both callers use it.
#
# Prints a COUNT to stdout:
#   >0  a real verdict (approve / request_changes / comment) already stands
#       on this head — an INFRA failure must stand down rather than demote it.
#    0  no such verdict — publish the blocking check exactly as before.
#
# Fail-closed by construction: every failure mode below (API error, malformed
# JSON, missing output, in-progress run) yields 0, i.e. "publish and block".
# Only positive evidence of a completed non-error verdict suppresses.
#
# Deliberately NOT suppressed for real verdicts: approve / request_changes /
# comment are judgements about the code and must be able to supersede each
# other in either direction, including approve -> request_changes. Only
# verdicts that judged nothing defer.
#
# Usage: standing-verdict.sh <owner/repo> <head_sha>
#
# errexit is safe here despite every failure path needing to yield 0: the one
# command that can fail (gh api) is guarded by `|| COUNT=0`, which suppresses
# errexit for that pipeline, and the `case` below normalises anything
# non-numeric. Callers additionally wrap the invocation in `|| echo 0`.
set -euo pipefail

REPO_FULL="${1:?usage: standing-verdict.sh <owner/repo> <head_sha>}"
HEAD_SHA="${2:?usage: standing-verdict.sh <owner/repo> <head_sha>}"

# The verdict is embedded as a marker comment in the check-run output. The
# publish step only ever writes `output.summary` (including its oversized-body
# truncation branch) — `output.text` is never set anywhere in this workflow —
# but both fields are read here defensively in case a future publisher uses
# `output.text` instead.
COUNT=$(gh api "repos/$REPO_FULL/commits/$HEAD_SHA/check-runs" \
  --jq '[.check_runs[]
         | select(.name == "cog-review" and .status == "completed")
         | (.output.summary // "") + (.output.text // "")]
        | map(select(test("cog-review:v1")))
        | map(capture("\"verdict\":\"(?<v>[a-z_]+)\"").v)
        | map(select(. != "error"))
        | length' 2>/dev/null) || COUNT=0

# Guard against a non-numeric result (empty output, jq error, API hiccup):
# anything that is not a plain integer means "no evidence", which is 0.
case "$COUNT" in
  ''|*[!0-9]*) COUNT=0 ;;
esac

printf '%s' "$COUNT"
