#!/usr/bin/env bash
# Synchronize source files over SSH. Preview unless --apply is explicit.
set -euo pipefail

usage() {
    cat <<'HELP'
Usage: bash scripts/sync_code.sh [--apply] [--mirror] [--dest USER@HOST:/absolute/project/]

Default: preview incremental updates to
  YOUR_USER@YOUR_SERVER:/home/YOUR_USER/PROJECT_EfficientFlow/

  --apply    Transfer the previewed updates (otherwise --dry-run).
  --mirror   Also remove remote files absent locally, except excluded paths.
             Preview this mode first. Remote-only source files may be removed.
  --dest     Override the destination for reuse with another server/project.
  --help     Show this help.

Source: the directory containing this script's parent (not the current shell).
Exclusions: .rsyncignore in that source directory.
Requires rsync locally and remotely, and an SSH connection.
Use RSYNC_RSH='ssh -p 2222' for a different SSH port.
HELP
}

source_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
destination='YOUR_USER@YOUR_SERVER:/home/YOUR_USER/PROJECT_EfficientFlow/'
apply=false
mirror=false
while (($#)); do
    case "$1" in
        --apply) apply=true; shift ;;
        --mirror) mirror=true; shift ;;
        --dest)
            if (($# < 2)); then printf '%s\n' 'Missing value after --dest' >&2; exit 2; fi
            destination="$2"; shift 2 ;;
        --help|-h) usage; exit 0 ;;
        *) printf 'Unknown argument: %s\n' "$1" >&2; usage >&2; exit 2 ;;
    esac
done

if [[ ! "$destination" =~ ^([a-zA-Z0-9_.-]+@)?[a-zA-Z0-9_.-]+:/ ]]; then
    printf '%s\n' '--dest must use HOST:/absolute/project/ or USER@HOST:/absolute/project/.' >&2
    exit 2
fi
remote_path="${destination#*:}"
remote_path="${remote_path%/}"
if [[ -z "$remote_path" || "$remote_path" == / || "$remote_path" == /home || "$remote_path" == /root ]]; then
    printf '%s\n' 'Refusing a filesystem root or a broad home-directory destination.' >&2
    exit 2
fi
if [[ "$source_root" == / || ! -f "$source_root/.rsyncignore" ]]; then
    printf '%s\n' 'Source must be a project directory containing .rsyncignore; refusing to synchronize.' >&2
    exit 2
fi
if ! command -v rsync >/dev/null 2>&1; then
    printf '%s\n' 'rsync is missing. In MSYS2: pacman -S --needed rsync openssh' >&2
    exit 127
fi

# Content checksums avoid retransferring unchanged files with different mtimes.
# Preserve times/symlinks, without imposing Windows ownership/permission bits.
args=(-rltzc --omit-dir-times --safe-links --itemize-changes --human-readable
      --stats --exclude-from="$source_root/.rsyncignore")
if [[ "$apply" == false ]]; then args+=(--dry-run); fi
if [[ "$mirror" == true ]]; then args+=(--delete-delay); fi

printf 'Source: %s/\nDestination: %s/\n' "$source_root" "$remote_path"
printf 'SSH target: %s\n' "${destination%%:*}"
printf 'Apply: %s; mirror deletions: %s\n' "$apply" "$mirror"
if [[ "$apply" == false ]]; then printf '%s\n' 'PREVIEW ONLY: no remote files will be changed.'; fi
if [[ "$mirror" == true ]]; then printf '%s\n' 'MIRROR: remote-only, non-excluded files are eligible for deletion.'; fi

# The trailing slash copies the CONTENTS, avoiding another EfficientFlow/ level.
rsync "${args[@]}" -- "$source_root/" "${destination%/}/"
