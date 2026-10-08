#!/usr/bin/env bash
# Render every chapter and publish the website to the gh-pages branch:
#   https://philippevelha.github.io/Optoelectronics-teaching/<chapter>/
# Usage (from the repository root):  bash publish.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
CHAPTERS="chapter2"
SITE="$ROOT/.gh-pages"

for ch in $CHAPTERS; do
  echo "== rendering $ch"
  (cd "$ROOT/$ch" && quarto render)
done

cd "$ROOT"
git fetch origin gh-pages 2>/dev/null || true
rm -rf "$SITE"; git worktree prune
if git show-ref --quiet refs/remotes/origin/gh-pages; then
  git worktree add -B gh-pages "$SITE" origin/gh-pages
else
  git worktree add --detach "$SITE"
  (cd "$SITE" && git checkout --orphan gh-pages && git rm -rfq . )
fi

cp "$ROOT/site/index.html" "$SITE/index.html"
touch "$SITE/.nojekyll"
for ch in $CHAPTERS; do
  rm -rf "$SITE/$ch"
  cp -r "$ROOT/$ch/_book" "$SITE/$ch"
done

cd "$SITE"
git add -A
if git diff --cached --quiet; then
  echo "== website already up to date"
else
  git commit -qm "Publish website $(date +%Y-%m-%d)"
  git push -u origin gh-pages
  echo "== published: https://philippevelha.github.io/Optoelectronics-teaching/"
fi
cd "$ROOT"; git worktree remove --force "$SITE"
