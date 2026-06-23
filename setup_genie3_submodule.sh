#!/usr/bin/env bash
# Convert the local external/genie3 symlink into a proper pinned git submodule.
#
# Could not be done at build time: this NERSC node has no GitHub network access
# ("could not read Username for 'https://github.com'"), so the offline scaffold uses a
# symlink to the local working checkout (~/genie3) instead.  Run this on a machine WITH
# GitHub access to make external/genie3 a reproducible, pinned submodule.
set -euo pipefail

REMOTE="$(cat external/genie3_remote_url.txt)"
PIN="$(cat external/genie3_pinned_commit.txt)"

echo "remote: $REMOTE"
echo "pin:    $PIN"

# remove the symlink stand-in
rm -f external/genie3

git submodule add "$REMOTE" external/genie3
git -C external/genie3 fetch --all
git -C external/genie3 checkout "$PIN"
git add .gitmodules external/genie3
echo "Done. Commit the submodule pin with:  git commit -m 'vendor genie3 as pinned submodule'"
