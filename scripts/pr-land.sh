#!/usr/bin/env bash
# pr-land.sh: merge a PR only when the standing authorization holds.
#
# Authorization (Chaz, 2026-10-04): the agent may merge its own PR when
#   1. every CI check on the current head has passed, AND
#   2. the independent cog-review gate APPROVED that exact head with no
#      requested changes.
# Anything else (pending, failed, changes requested, a comment verdict,
# reviewer error, head moved) is not a merge. The script never bypasses or
# overrides branch protection.
#
# Decisions use only server-set signals: check-run conclusions, the bot's
# APPROVED review pinned to the head SHA (via pr-await-review.sh), and
# GitHub's own mergeStateStatus. The review's findings count is parsed from
# the untrusted marker, so it can only make this script MORE conservative
# (findings > 0 blocks); it can never authorize a merge on its own.
#
# Usage: pr-land.sh <owner/repo> <pr-number> [timeout-seconds] [--dry-run]
# Exit:  0 merged (or would merge, with --dry-run) | 2 changes requested
#        3 timeout | 4 escalate (not authorized; see stderr) | 5 CI failed

set -euo pipefail
REPO="${1:?usage: pr-land.sh <owner/repo> <pr> [timeout] [--dry-run]}"
PR="${2:?usage: pr-land.sh <owner/repo> <pr> [timeout] [--dry-run]}"
TIMEOUT="${3:-1800}"
DRY=0
for a in "$@"; do [ "$a" = "--dry-run" ] && DRY=1; done
[ "$TIMEOUT" = "--dry-run" ] && TIMEOUT=1800
HERE="$(cd "$(dirname "$0")" && pwd)"
POLL="${PR_LAND_POLL_INTERVAL:-20}"

# 1. Independent review: blocks until the gate concludes on the current head.
set +e
VERDICT=$(bash "$HERE/pr-await-review.sh" "$REPO" "$PR" "$TIMEOUT")
RC=$?
set -e
printf '%s\n' "$VERDICT"
if [ "$RC" -ne 0 ]; then
  echo "pr-land: review gate did not approve (exit $RC); not merging" >&2
  exit "$RC"
fi
HEAD_SHA=$(printf '%s' "$VERDICT" | jq -r '.head_sha')
FINDINGS=$(printf '%s' "$VERDICT" | jq -r '(.findings // []) | length')
if [ "$FINDINGS" != "0" ]; then
  echo "pr-land: approved, but with $FINDINGS finding(s); 'no changes' not met; escalating" >&2
  exit 4
fi

# 2. All CI green on the same head. Wait for anything still running.
deadline=$(( $(date +%s) + TIMEOUT ))
while :; do
  # One row per check name: the LATEST run (highest id) supersedes earlier
  # rows for the same name on this SHA. Without this, a flaky cog-review
  # failure that a later run on the same head replaced with an approve would
  # block forever (cog-review finding on #156).
  RUNS=$(gh api "repos/$REPO/commits/$HEAD_SHA/check-runs?per_page=100" --paginate \
    --jq '.check_runs[] | {id, name, status, conclusion}' |
    jq -s 'group_by(.name) | map(max_by(.id))')
  PENDING=$(printf '%s' "$RUNS" | jq '[.[] | select(.status != "completed")] | length')
  BAD=$(printf '%s' "$RUNS" | jq -r '[.[] | select(.status == "completed" and (.conclusion | IN("success","skipped","neutral") | not)) | "\(.name)=\(.conclusion)"] | join(", ")')
  TOTAL=$(printf '%s' "$RUNS" | jq 'length')
  # Don't call CI green before the workflows have registered their checks:
  # require the gate's own check and every check name branch protection
  # requires to be present (not a bare row count).
  REQUIRED=$(gh api "repos/$REPO/branches/$(gh pr view "$PR" -R "$REPO" --json baseRefName --jq .baseRefName)/protection/required_status_checks" \
    --jq '.contexts' 2>/dev/null || echo '[]')
  MISSING=$(printf '%s' "$RUNS" | jq -r --argjson req "$REQUIRED" \
    '. as $runs | [($req + ["cog-review"]) | unique[] | select(. as $n | ($runs | map(.name) | index($n)) == null)] | join(", ")')
  if [ -n "$BAD" ]; then
    echo "pr-land: CI not green on ${HEAD_SHA:0:7}: $BAD" >&2
    exit 5
  fi
  [ "$PENDING" = "0" ] && [ -z "$MISSING" ] && break
  [ -n "$MISSING" ] && echo "pr-land: waiting for check(s) to appear: $MISSING" >&2
  if [ "$(date +%s)" -ge "$deadline" ]; then
    echo "pr-land: CI still pending past ${TIMEOUT}s" >&2
    exit 3
  fi
  echo "pr-land: $PENDING check(s) still running on ${HEAD_SHA:0:7}" >&2
  sleep "$POLL"
done

# 3. GitHub agrees it is mergeable at this head (branch protection satisfied).
STATE=$(gh pr view "$PR" -R "$REPO" --json headRefOid,mergeStateStatus,isDraft,state \
  --jq '"\(.headRefOid) \(.mergeStateStatus) \(.isDraft) \(.state)"')
read -r CUR_SHA MERGE_STATE DRAFT PR_STATE <<<"$STATE"
if [ "$CUR_SHA" != "$HEAD_SHA" ]; then
  echo "pr-land: head moved ${HEAD_SHA:0:7} -> ${CUR_SHA:0:7}; not merging" >&2
  exit 4
fi
if [ "$PR_STATE" != "OPEN" ] || [ "$DRAFT" = "true" ] || [ "$MERGE_STATE" != "CLEAN" ]; then
  echo "pr-land: not mergeable (state=$PR_STATE draft=$DRAFT mergeState=$MERGE_STATE)" >&2
  exit 4
fi

echo "pr-land: authorized: $TOTAL checks green + cog-review APPROVED (0 findings) @ ${HEAD_SHA:0:7}" >&2
if [ "$DRY" = "1" ]; then
  echo "pr-land: --dry-run, not merging" >&2
  exit 0
fi
# --match-head-commit: GitHub refuses if the head moved after our checks.
gh pr merge "$PR" -R "$REPO" --squash --delete-branch --match-head-commit "$HEAD_SHA"
echo "pr-land: merged $REPO#$PR @ ${HEAD_SHA:0:7}" >&2
