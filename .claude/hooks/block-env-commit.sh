#!/usr/bin/env bash
# PreToolUse hook on the Bash tool: blocks commands that look like they
# `git add` a .env file. This repo already had a hardcoded API key
# committed once this session (see Chat_Bot/test.py's history) - .env is
# gitignored on purpose, so this is a learning example of catching that
# mistake automatically instead of relying on remembering not to.
#
# Deliberately checks `git add` only, not `git commit`: a commit MESSAGE
# routinely mentions ".env" in plain English (e.g. "update .env.example
# docs") with no file actually staged, and scanning commit text for that
# produces false positives on totally unrelated commits. `git add` is
# where a literal .env path argument would actually appear.
#
# Simple, not bulletproof: it matches the literal text of the command, so
# it won't catch e.g. `git add -A` silently picking up a .env file without
# naming it. That tradeoff is fine for a learning example, not a
# production security control.

cmd=$(python -c "import json,sys; print(json.load(sys.stdin).get('tool_input', {}).get('command', ''))")

# Isolate just the "git add ..." clause(s) - up to the next &&/;/| or end of
# line - so a .env mention elsewhere in a compound command (e.g. a later
# `git commit -m "..."` whose MESSAGE happens to say ".env") can't trigger
# this. Only text actually passed to `git add` counts.
add_clauses=$(echo "$cmd" | grep -oE 'git[[:space:]]+add\b[^&;|]*')

if echo "$add_clauses" | grep -qE '(^|[^.[:alnum:]_-])\.env([^.[:alnum:]_-]|$)' \
   && ! echo "$add_clauses" | grep -qE '\.env\.example'; then
  cat <<'EOF'
{"hookSpecificOutput": {"hookEventName": "PreToolUse", "permissionDecision": "deny", "permissionDecisionReason": "Blocked: this command looks like it stages or commits a .env file. .env is gitignored on purpose - this repo already had a hardcoded API key committed once this session. If you need to share a template, commit .env.example (blank values only) instead."}}
EOF
fi

exit 0
