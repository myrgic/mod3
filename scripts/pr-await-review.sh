#!/usr/bin/env bash
# pr-await-review.sh — block until a PR's cog-review gate concludes, emit the
# machine-readable verdict, exit-code the reaction.
#
# The repo-side gate (pr-review.yml) submits one GitHub review (APPROVE /
# REQUEST_CHANGES / COMMENT) carrying a `<!-- cog-review:v1 {json} -->` marker
# and one `cog-review` check-run per head commit. This script is the local
# workflow's side of the handshake:
# file PR -> pr-await-review -> react (fix+push / merge / escalate).
# The gate can approve; merging remains the caller's act.
#
# TRUST MODEL (security review 2026-07-05, findings #1/#2): the gate DECISION
# is driven only by GitHub server-set signals — the `cog-review` check-run
# `conclusion` and, for approve, a review whose `.state == APPROVED`, `.user`
# is github-actions[bot], and `.commit_id` equals the head we polled. The
# `<!-- cog-review:v1 ... -->` marker is UNTRUSTED display data (the model can
# echo a lookalike into its own findings text); it is parsed only to surface
# findings to the operator and NEVER decides the exit code. Auto-merge
# eligibility (exit 0) requires BOTH check-run success AND a matching bot
# APPROVED review — if the check says success but no genuine approval exists
# (e.g. the repo's approve-setting is off and the gate fell back to a
# comment), we escalate rather than merge.
#
# Usage: pr-await-review.sh <owner/repo> <pr-number> [timeout-seconds]
#
# Exit codes (the reaction contract):
#   0  approve          -> proceed to merge (merge authority stays local)
#   2  request_changes  -> reconcile findings, push, re-await (cap rounds!)
#   4  comment/error/    -> gate concluded without a genuine approval; or the
#      no-genuine-approve    head moved under us; escalate to operator
#   3  timeout          -> reviewer silence; fail closed, escalate to operator
#
# stdout: one JSON object
#   {conclusion, verdict, decision, head_sha, findings, unverified_notes, review_url}
# All progress chatter goes to stderr.

set -euo pipefail

REPO="${1:?usage: pr-await-review.sh <owner/repo> <pr-number> [timeout-seconds]}"
PR="${2:?usage: pr-await-review.sh <owner/repo> <pr-number> [timeout-seconds]}"
TIMEOUT="${3:-900}"
POLL_INTERVAL="${PR_AWAIT_POLL_INTERVAL:-20}"
BOT_LOGIN="${PR_AWAIT_BOT_LOGIN:-github-actions[bot]}"

emit() { # emit <conclusion> <verdict> <decision> <exit> [verdict_json]
  local vjson="${5:-{\}}"
  printf '%s' "$vjson" | jq -c \
    --arg conclusion "$1" --arg verdict "$2" --arg decision "$3" \
    --arg sha "$HEAD_SHA" --arg url "${REVIEW_URL:-null}" \
    '{conclusion:$conclusion, verdict:$verdict, decision:$decision, head_sha:$sha,
      findings:(.findings // []), unverified_notes:(.unverified_notes // []),
      review_url:(if $url == "null" then null else $url end)}'
  exit "$4"
}

HEAD_SHA=$(gh api "repos/$REPO/pulls/$PR" --jq '.head.sha')
REVIEW_URL="null"
echo "awaiting cog-review on $REPO#$PR @ ${HEAD_SHA:0:7} (timeout ${TIMEOUT}s)" >&2

deadline=$(( $(date +%s) + TIMEOUT ))
while :; do
  # The gate re-runs per push; only trust a check-run for the CURRENT head.
  RUN=$(gh api "repos/$REPO/commits/$HEAD_SHA/check-runs?check_name=cog-review" \
    --jq '[.check_runs[]] | sort_by(.completed_at) | last // empty' 2>/dev/null || true)

  if [ -n "$RUN" ] && [ "$(printf '%s' "$RUN" | jq -r '.status')" = "completed" ]; then
    CONCLUSION=$(printf '%s' "$RUN" | jq -r '.conclusion')
    break
  fi

  # A push while we waited moves the goalposts; track the new head.
  NEW_HEAD=$(gh api "repos/$REPO/pulls/$PR" --jq '.head.sha')
  if [ "$NEW_HEAD" != "$HEAD_SHA" ]; then
    HEAD_SHA="$NEW_HEAD"
    echo "head moved to ${HEAD_SHA:0:7}; re-awaiting" >&2
  fi

  if [ "$(date +%s)" -ge "$deadline" ]; then
    echo "cog-review silent past ${TIMEOUT}s — failing closed" >&2
    emit "timeout" "timeout" "escalate" 3
  fi
  sleep "$POLL_INTERVAL"
done

# --- Genuine-approval check (server-set, unforgeable): a review by the bot,
# state APPROVED, pinned to the exact head we concluded on. ---
APPROVED_REVIEW=$(gh api "repos/$REPO/pulls/$PR/reviews" --paginate \
  --jq "[.[] | select(.user.login == \"$BOT_LOGIN\" and .state == \"APPROVED\" and .commit_id == \"$HEAD_SHA\")] | last // empty")

# --- Changes-requested check (server-set): a bot CHANGES_REQUESTED review at
# head. The gate maps request_changes AND comment/error/canary-fail/no-token
# ALL to check-run conclusion=failure (fail closed), so `failure` alone can no
# longer distinguish "real findings to fix" (reconcile) from "no actionable
# verdict" (escalate). The review STATE — also server-set and unforgeable —
# does: CHANGES_REQUESTED means genuine findings; a COMMENTED review or NO
# review (errors post none) means escalate to a human. ---
CHANGES_REQUESTED_REVIEW=$(gh api "repos/$REPO/pulls/$PR/reviews" --paginate \
  --jq "[.[] | select(.user.login == \"$BOT_LOGIN\" and .state == \"CHANGES_REQUESTED\" and .commit_id == \"$HEAD_SHA\")] | last // empty")

# --- Marker extraction is DISPLAY-ONLY (untrusted). The real marker is the
# LAST line of the gate's body/summary (the workflow appends it after all
# model-authored text), so take the last match, not the first — an injected
# lookalike can only appear earlier, inside findings. ---
extract_marker() { # stdin: text -> stdout: last cog-review marker json (or empty)
  sed -n 's/.*<!-- cog-review:v1 \(.*\) -->.*/\1/p' | tail -1
}
VERDICT_JSON=""
# Prefer the bot review body for findings; fall back to the check-run summary.
BOT_REVIEW=$(gh api "repos/$REPO/pulls/$PR/reviews" --paginate \
  --jq "[.[] | select(.user.login == \"$BOT_LOGIN\" and .commit_id == \"$HEAD_SHA\")] | last // empty")
if [ -n "$BOT_REVIEW" ]; then
  REVIEW_URL=$(printf '%s' "$BOT_REVIEW" | jq -r '.html_url')
  VERDICT_JSON=$(printf '%s' "$BOT_REVIEW" | jq -r '.body' | extract_marker)
fi
if [ -z "$VERDICT_JSON" ]; then
  VERDICT_JSON=$(printf '%s' "$RUN" | jq -r '.output.summary // ""' | extract_marker)
fi
# Validate it parses; a lookalike that isn't valid JSON is discarded silently.
if ! printf '%s' "$VERDICT_JSON" | jq -e . >/dev/null 2>&1; then
  VERDICT_JSON="{}"
fi
# The marker's self-reported head must match; otherwise it's stale/planted.
MARKER_SHA=$(printf '%s' "$VERDICT_JSON" | jq -r '.head_sha // empty' 2>/dev/null || true)
if [ -n "$MARKER_SHA" ] && [ "$MARKER_SHA" != "$HEAD_SHA" ]; then
  VERDICT_JSON="{}"
fi
# verdict shown to the operator is display-only; the check-run conclusion is
# authoritative for the decision below.
VERDICT=$(printf '%s' "$VERDICT_JSON" | jq -r '.verdict // empty' 2>/dev/null || true)
[ -n "$VERDICT" ] || VERDICT="$CONCLUSION"

# --- TOCTOU guard: re-confirm the PR head hasn't advanced since we concluded.
# A push between conclusion and here means our approval no longer describes the
# tip — never merge on it. ---
CURRENT_HEAD=$(gh api "repos/$REPO/pulls/$PR" --jq '.head.sha')
if [ "$CURRENT_HEAD" != "$HEAD_SHA" ]; then
  echo "head advanced ${HEAD_SHA:0:7} -> ${CURRENT_HEAD:0:7} after conclusion; not merging" >&2
  HEAD_SHA="$CURRENT_HEAD"
  emit "$CONCLUSION" "$VERDICT" "escalate" 4 "$VERDICT_JSON"
fi

# --- Decision: server-set signals only. ---
case "$CONCLUSION" in
  success)
    if [ -n "$APPROVED_REVIEW" ]; then
      echo "cog-review: success + genuine bot APPROVED review @ ${HEAD_SHA:0:7}" >&2
      emit "$CONCLUSION" "$VERDICT" "merge" 0 "$VERDICT_JSON"
    fi
    echo "check-run success but NO genuine bot approval at head (degraded/comment mode); escalating" >&2
    emit "$CONCLUSION" "$VERDICT" "escalate" 4 "$VERDICT_JSON"
    ;;
  failure)
    # Only a genuine CHANGES_REQUESTED review means "findings to fix".
    # comment / error / canary-fail / no-token also land here (fail closed)
    # but carry no actionable findings — escalate to a human instead of
    # telling the caller to push phantom fixes.
    if [ -n "$CHANGES_REQUESTED_REVIEW" ]; then
      echo "cog-review: request_changes (bot CHANGES_REQUESTED @ ${HEAD_SHA:0:7})" >&2
      emit "$CONCLUSION" "$VERDICT" "reconcile" 2 "$VERDICT_JSON"
    fi
    echo "cog-review: blocking failure with NO changes-requested review (comment/error/canary/no-token) @ ${HEAD_SHA:0:7}; escalating" >&2
    emit "$CONCLUSION" "$VERDICT" "escalate" 4 "$VERDICT_JSON"
    ;;
  *)
    emit "$CONCLUSION" "$VERDICT" "escalate" 4 "$VERDICT_JSON"
    ;;
esac
