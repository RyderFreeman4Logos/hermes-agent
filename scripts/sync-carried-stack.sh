#!/usr/bin/env bash
set -euo pipefail

# Sync a carried commit stack onto origin/main with a quiet success path.
#
# Usage: scripts/sync-carried-stack.sh [--check]
#   --check  Rehearse every replay as a three-way merge without fetching or
#            updating refs. Emit JSON only if a conflict is expected.
#
# A successful real sync writes only refs/heads/local/hermes-carried-stack-candidate.
# It never mutates the current carried branch, pushes, or operates on the live
# ~/.hermes/hermes-agent checkout.

usage() {
  printf 'Usage: %s [--check]\n' "${0##*/}" >&2
}

die() {
  printf 'sync-carried-stack: %s\n' "$*" >&2
  exit 2
}

json_string() {
  local value=$1
  value=${value//\\/\\\\}
  value=${value//\"/\\\"}
  value=${value//$'\b'/\\b}
  value=${value//$'\f'/\\f}
  value=${value//$'\n'/\\n}
  value=${value//$'\r'/\\r}
  value=${value//$'\t'/\\t}
  printf '"%s"' "$value"
}

emit_conflict() {
  local mode=$1
  local commit=$2
  local subject=$3
  shift 3
  local -a files=("$@")
  local index

  printf '{\n'
  printf '  "status": "conflict",\n'
  printf '  "mode": '
  json_string "$mode"
  printf ',\n'
  printf '  "stack_branch": '
  json_string "$stack_branch"
  printf ',\n'
  printf '  "old_base": '
  json_string "$old_base"
  printf ',\n'
  printf '  "origin_main": '
  json_string "$origin_main"
  printf ',\n'
  printf '  "commit": '
  json_string "$commit"
  printf ',\n'
  printf '  "commit_subject": '
  json_string "$subject"
  printf ',\n'
  printf '  "conflicted_files": ['
  for index in "${!files[@]}"; do
    if (( index > 0 )); then
      printf ', '
    fi
    json_string "${files[index]}"
  done
  printf ']\n}\n'
}

assert_safe_checkout() {
  local status entry
  local -a unexpected=()

  if [[ -e "$HOME/.hermes/hermes-agent" ]]; then
    local live_root
    live_root=$(cd "$HOME/.hermes/hermes-agent" && pwd -P) || die 'cannot resolve live checkout path'
    [[ "$repo_root" != "$live_root" ]] || die 'refusing to operate on the live ~/.hermes/hermes-agent checkout'
  fi

  while IFS= read -r -d '' entry; do
    # The only deliberately tolerated dirty path is the pre-existing, untracked
    # fork-main helper. Any staged, modified, deleted, renamed, or other
    # untracked path can make a rebase unsafe.
    [[ "$entry" == '?? scripts/sync-fork-main.sh' ]] || unexpected+=("${entry:3}")
  done < <(git status --porcelain=v1 -z --untracked-files=all)

  if (( ${#unexpected[@]} > 0 )); then
    printf 'sync-carried-stack: refusing dirty worktree (allowed only: untracked scripts/sync-fork-main.sh):\n' >&2
    printf '  %s\n' "${unexpected[@]}" >&2
    exit 2
  fi
}

make_simulated_commit() {
  local tree=$1
  local parent=$2
  GIT_AUTHOR_NAME='sync-carried-stack' \
  GIT_AUTHOR_EMAIL='sync-carried-stack@invalid' \
  GIT_COMMITTER_NAME='sync-carried-stack' \
  GIT_COMMITTER_EMAIL='sync-carried-stack@invalid' \
    git commit-tree "$tree" -p "$parent" -m 'sync-carried-stack preflight'
}

# Rehearse the rebase one commit at a time without changing a ref. A simulated
# commit has the replayed tree but the original commit as parent, making the
# next merge-base exactly that next commit's parent.
preflight_rebase() {
  local commit merge_output merge_status merged_tree line
  local simulated_head

  conflict_commit=''
  conflict_subject=''
  conflict_files=()
  simulated_head=$(make_simulated_commit "${origin_main}^{tree}" "$old_base") || die 'cannot initialize preflight merge state'

  while IFS= read -r commit; do
    [[ -n "$commit" ]] || continue
    set +e
    merge_output=$(git merge-tree --write-tree --name-only "$simulated_head" "$commit" 2>&1)
    merge_status=$?
    set -e

    if (( merge_status == 0 )); then
      merged_tree=${merge_output%%$'\n'*}
      [[ "$merged_tree" =~ ^[0-9a-f]{40}$ ]] || die "unexpected merge-tree result while replaying $commit"
      simulated_head=$(make_simulated_commit "$merged_tree" "$commit") || die "cannot advance preflight merge state for $commit"
      continue
    fi

    if (( merge_status != 1 )); then
      printf '%s\n' "$merge_output" >&2
      die "merge rehearsal failed while replaying $commit"
    fi

    conflict_commit=$commit
    conflict_subject=$(git show -s --format=%s "$commit")
    while IFS= read -r line; do
      [[ -z "$line" || "$line" =~ ^[0-9a-f]{40}$ ]] && continue
      conflict_files+=("$line")
    done <<<"$merge_output"
    return 1
  done < <(git rev-list --reverse "$old_base..$stack_branch")

  return 0
}

check_mode=false
case ${1:-} in
  '') ;;
  --check) check_mode=true ;;
  -h|--help)
    usage
    exit 0
    ;;
  *)
    usage
    exit 2
    ;;
esac

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
repo_root=$(git -C "$script_dir" rev-parse --show-toplevel) || die 'script must run from a Git checkout'
cd "$repo_root"
assert_safe_checkout

stack_branch=$(git symbolic-ref --quiet --short HEAD) || die 'refusing detached HEAD; check out the carried stack branch first'
git rev-parse --verify origin/main >/dev/null 2>&1 || die 'origin/main is unavailable'

if [[ "$check_mode" == false ]]; then
  git fetch origin --quiet
fi

origin_main=$(git rev-parse --verify origin/main)
old_base=$(git merge-base "$stack_branch" origin/main) || die 'cannot find a merge-base between current stack and origin/main'
stack_tip=$(git rev-parse --verify "$stack_branch")

if ! preflight_rebase; then
  emit_conflict "$([[ "$check_mode" == true ]] && printf check || printf sync)" \
    "$conflict_commit" "$conflict_subject" "${conflict_files[@]}"
  exit 1
fi

if [[ "$check_mode" == true ]]; then
  exit 0
fi

candidate_ref='refs/heads/local/hermes-carried-stack-candidate'
if [[ "$old_base" == "$origin_main" ]]; then
  git update-ref "$candidate_ref" "$stack_tip"
  exit 0
fi

# Rebase a disposable copy of the stack. It begins at the carried tip so that
# `git rebase --onto origin/main <old-base>` can replay it, and ends based on
# origin/main. The user's current stack branch is never rewritten.
temp_branch="local/sync-carried-stack-tmp-${BASHPID}-${RANDOM}"
git branch "$temp_branch" "$stack_tip"

cleanup_temp_branch() {
  local current_branch
  if git rev-parse --verify REBASE_HEAD >/dev/null 2>&1; then
    git rebase --abort >/dev/null 2>&1 || true
  fi
  current_branch=$(git symbolic-ref --quiet --short HEAD || true)
  if [[ "$current_branch" != "$stack_branch" ]]; then
    git switch --quiet "$stack_branch" >/dev/null 2>&1 || true
  fi
  if [[ -n "${temp_branch:-}" ]]; then
    git branch -D "$temp_branch" >/dev/null 2>&1 || true
  fi
}
trap cleanup_temp_branch EXIT

git switch --quiet "$temp_branch"
set +e
rebase_output=$(git rebase --onto origin/main "$old_base" 2>&1)
rebase_status=$?
set -e

if (( rebase_status != 0 )); then
  rebase_head=$(git rev-parse --verify REBASE_HEAD 2>/dev/null || true)
  mapfile -t actual_conflict_files < <(git diff --name-only --diff-filter=U)
  cleanup_temp_branch
  temp_branch=''
  trap - EXIT

  if [[ -n "$rebase_head" && ${#actual_conflict_files[@]} -gt 0 ]]; then
    emit_conflict sync "$rebase_head" "$(git show -s --format=%s "$rebase_head")" "${actual_conflict_files[@]}"
    exit 1
  fi

  printf '%s\n' "$rebase_output" >&2
  die 'rebase failed before producing a conflict report'
fi

candidate_tip=$(git rev-parse --verify HEAD)
git switch --quiet "$stack_branch"
git branch -D "$temp_branch" >/dev/null
temp_branch=''
trap - EXIT
git update-ref "$candidate_ref" "$candidate_tip"
