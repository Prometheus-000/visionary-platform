"""What the two validator bugs cost, measured on real model output.

The sweep says the proposed change admits nothing it should refuse. This says
what it *buys*, which is the other half and the one that decides whether the
fix is worth making: how much ordinary prose the shipped checks throw away.
"""
import json, sys
sys.path.insert(0, "/Users/kenanalsarabi/visionary_platform/tools")
sys.argv = ["x"]
import smoke_parse as S
G = S.G


def preserved_fixed(modules, prose):
    """Consumption dropped. Coverage is a set union, so it never bought anything."""
    hay = prose.lower()
    if len(hay) != len(prose):
        hay = prose
    covered = []
    def claim(run):
        i = hay.find(run)
        if i < 0:
            return False
        covered.append((i, i + len(run)))
        return True
    for module in G["_walk_document"](modules):
        if module.get("role") == "subject" and module.get("origin") == "invented":
            return "invented a subject", covered
        for run in G["_derived_runs"](module):
            if not claim(run.lower()):
                return f"derived text not in the prose: {run[:60]!r}", covered
    return None, covered


def trust(modules, prose, preserved):
    if not modules:
        return "no elements"
    prose = G["_oneline"](prose)
    if not prose:
        return None
    reason, covered = preserved(modules, prose)
    if reason:
        return reason
    if G["_invention_share"](modules) > G["INVENTION_CEILING"]:
        return f"invention over ceiling"
    solid = sum(1 for c in prose if not c.isspace())
    if solid:
        seen = set()
        for a, b in covered:
            seen.update(range(a, b))
        if sum(1 for i in seen if not prose[i].isspace()) / solid < G["COVERAGE_FLOOR"]:
            return "coverage under floor"
    return None


docs = []
for line in open(sys.argv[1] if len(sys.argv) > 1 else
                 "/private/tmp/claude-501/-Users-kenanalsarabi-visionary-platform/"
                 "d2b0d3ba-ac67-42e7-aee9-eaf5f417dca3/scratchpad/dump_4b.log"):
    if line.startswith("DOC "):
        docs.append(json.loads(line[4:]))

print(f"{len(docs)} fragments of real model output\n")
ship_ok = fix_ok = 0
reasons_ship, reasons_fix = {}, {}
recovered = []
for d in docs:
    if d.get("error"):
        continue
    mods, prose = d["elements"], d["prose"]
    a = trust(mods, prose, G["_preserved"])
    b = trust(mods, prose, preserved_fixed)
    ship_ok += a is None
    fix_ok += b is None
    key = lambda r: (r or "TRUSTED").split(":")[0]
    reasons_ship[key(a)] = reasons_ship.get(key(a), 0) + 1
    reasons_fix[key(b)] = reasons_fix.get(key(b), 0) + 1
    if a is not None and b is None:
        recovered.append(prose)

n = sum(1 for d in docs if not d.get("error"))
print(f"  trusted as shipped   {ship_ok}/{n}  ({100*ship_ok//n}%)")
print(f"  trusted with the fix {fix_ok}/{n}  ({100*fix_ok//n}%)\n")
print("  why they were dropped — as shipped")
for r, c in sorted(reasons_ship.items(), key=lambda x: -x[1]):
    print(f"    {c:3}  {r}")
print("\n  why they were dropped — with the fix")
for r, c in sorted(reasons_fix.items(), key=lambda x: -x[1]):
    print(f"    {c:3}  {r}")
if recovered:
    print(f"\n  recovered by the fix ({len(recovered)}):")
    for p in recovered:
        print(f"    {p[:74]}")

# And the fidelity question, which is separate from trust: of the documents that
# ARE trusted, how many compile back to the prose they came from.
print()
exact = sum(1 for d in docs if not d.get("error")
            and trust(d["elements"], d["prose"], preserved_fixed) is None
            and S.compile_prompt(d["elements"]) == G["_oneline"](d["prose"]))
print(f"  of the {fix_ok} trusted, {exact} compile back to the prose exactly")


# ── the thresholds, against real output ─────────────────────────────────────
#
# CLAUDE.md leaves this open in as many words: both bounds were swept over
# hand-written documents, "and it cannot say what this interpreter actually
# produces. Re-run the sweep against a served endpoint once weights are up, and
# move them if the real distribution sits elsewhere." This is that re-run.
print("\n\n=== the two bounds, measured on real output rather than by hand ===\n")
shares, covs = [], []
for d in docs:
    if d.get("error") or not d["elements"]:
        continue
    mods, prose = d["elements"], G["_oneline"](d["prose"])
    reason, covered = preserved_fixed(mods, prose)
    if reason:
        continue
    solid = sum(1 for c in prose if not c.isspace()) or 1
    seen = set()
    for a, b in covered:
        seen.update(range(a, b))
    shares.append((G["_invention_share"](mods), d["prose"]))
    covs.append((sum(1 for i in seen if not prose[i].isspace()) / solid, d["prose"]))

shares.sort(reverse=True)
covs.sort()
print(f"  invention share over {len(shares)} trusted documents")
print(f"    max {shares[0][0]:.1%}   median {shares[len(shares)//2][0]:.1%}   "
      f"min {shares[-1][0]:.1%}   ceiling {G['INVENTION_CEILING']:.0%}")
print("    the five most inventive:")
for s, p in shares[:5]:
    print(f"      {s:6.1%}  {p[:62]}")
over = [ (s,p) for s,p in shares if s > G["INVENTION_CEILING"] ]
print(f"    over the ceiling: {len(over)}")

print(f"\n  coverage over {len(covs)} trusted documents")
print(f"    min {covs[0][0]:.1%}   median {covs[len(covs)//2][0]:.1%}   "
      f"max {covs[-1][0]:.1%}   floor {G['COVERAGE_FLOOR']:.0%}")
print("    the five sparsest:")
for c, p in covs[:5]:
    print(f"      {c:6.1%}  {p[:62]}")
under = [ (c,p) for c,p in covs if c < G["COVERAGE_FLOOR"] ]
print(f"    under the floor: {len(under)}")
print("\n  A bound is right if the real distribution sits clear of it. A ceiling"
      "\n  the model never approaches is a check that has never fired; a floor the"
      "\n  model routinely sits on is a check that is dropping good documents.")
