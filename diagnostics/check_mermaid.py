"""Validate the README Mermaid block against GitHub's stricter parser.

GitHub uses a recent mermaid that rejects some shorthand the local
renderer tolerates (notably [(("label"))] cylinder nodes and |"label"|
edge text). This extracts the block, syncs dev/rda_architecture.mmd,
and flags constructs known to break on GitHub.
"""
import re
import sys

TICK = chr(96) * 3
s = open("README.md", encoding="utf-8").read()
m = re.search(re.escape(TICK) + r"mermaid\n(.*?)\n" + re.escape(TICK), s, re.S)
if not m:
    print("NO MERMAID BLOCK FOUND")
    sys.exit(1)
block = m.group(1)
open("dev/rda_architecture.mmd", "w", encoding="utf-8").write(block + "\n")
print("extracted %d lines from README" % len(block.split("\n")))

bad = []
for i, line in enumerate(block.split("\n"), 1):
    stripped = line.strip()
    if not stripped:
        continue
    # cylinder / subroutine / asymmetric / rhombus compound shapes
    if re.search(r'\[(\(|\[|\{|\/)', stripped) or re.search(r'(\)|\]\]|\}\])', stripped):
        bad.append((i, "shape syntax", stripped[:70]))
    # dashed edge WITH a label  -.->|"text"|
    if re.search(r'-\.->\s*\|', stripped):
        bad.append((i, "dashed edge w/ label", stripped[:70]))
    # any edge label at all |"text"|
    if re.search(r'\|[^|]*"', stripped):
        bad.append((i, "edge label", stripped[:70]))
    # ~./ html entities
    if "~/" in stripped:
        bad.append((i, "html entity", stripped[:70]))

for i, kind, txt in bad:
    print("  line %d [%s]: %s" % (i, kind, txt))
print("suspicious constructs: %d" % len(bad))

# Sanity: every referenced node should be defined before use where it matters.
defined = set(re.findall(r'^\s*([A-Z][A-Z0-9_]*)\s*[\[\(\{]', block, re.M))
defined |= set(re.findall(r'^\s*([A-Z][A-Z0-9_]*)\[\[', block, re.M))
defined |= set(re.findall(r'subgraph\s+([A-Z][A-Z0-9_]*)', block))
used = set(re.findall(r'([A-Z][A-Z0-9_]*)\s*[-=.]', block))
print("undefined refs:", sorted(used - defined) or "none")
sys.exit(1 if bad else 0)
