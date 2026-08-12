"""
Freeze what the page currently does, so a rewrite has something to fail against.

    /opt/homebrew/bin/python3.11 tools/preview_ui.py 8791 &
    python3.11 tools/ui-checks/baseline.py capture    # writes baseline/*.json
    python3.11 tools/ui-checks/baseline.py check      # re-runs and diffs

The React port is a rewrite of 4,544 lines of a page whose value is mostly in
decisions that are not inferable from the code they produced — the console
budget, the separator rule, which pills a silent model drops. Those survive a
rewrite only if something fails when they do not, and "someone looks at it"
is not that thing: every one of them looks fine.

So `capture` is run against the current page and the result is committed. From
then on `check` is the contract. A diff is not automatically a bug — it is a
question, and the answer belongs in the commit that changes the baseline.

Two probes, deliberately split by what they need:

  * `compile` talks HTTP only, so it runs anywhere and is the one that matters
    most — the compilers are staying exactly where they are, and the whole
    point of `/api/compile` is that the page never grows its own copy.
  * `console` needs Chrome, because the thing being pinned is a measured
    height and there is no way to measure it without laying the page out.
"""
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
STORE = HERE / "baseline"
URL = "http://localhost:8791"


def probes():
    """Imported lazily: `compile` must still run on a machine with no Chrome."""
    import probe_compile
    probes = {"compile": probe_compile.capture}
    try:
        import probe_console
        probes["console"] = probe_console.capture
    except ImportError as exc:                      # playwright absent
        print(f"  (skipping console probe: {exc})", file=sys.stderr)
    return probes


def reachable():
    import urllib.error
    import urllib.request
    try:
        with urllib.request.urlopen(f"{URL}/api/state", timeout=10):
            return True
    except (urllib.error.URLError, OSError):
        return False


def capture(names):
    STORE.mkdir(exist_ok=True)
    for name, fn in probes().items():
        if names and name not in names:
            continue
        print(f"capturing {name} …")
        data = fn()
        path = STORE / f"{name}.json"
        # sort_keys so a dict ordering change is never mistaken for drift.
        path.write_text(json.dumps(data, indent=1, sort_keys=True, ensure_ascii=False) + "\n")
        print(f"  wrote {path.relative_to(HERE.parent.parent)} "
              f"({len(data)} entries, {path.stat().st_size} bytes)")


def walk(node, trail=()):
    """Flatten to leaf paths so a diff points at the case, not at the blob."""
    if isinstance(node, dict):
        for k in sorted(node):
            yield from walk(node[k], trail + (str(k),))
    elif isinstance(node, list):
        for i, v in enumerate(node):
            yield from walk(v, trail + (str(i),))
    else:
        yield ".".join(trail), node


def check(names):
    failed = False
    for name, fn in probes().items():
        if names and name not in names:
            continue
        path = STORE / f"{name}.json"
        if not path.exists():
            print(f"{name}: no baseline — run `capture` first", file=sys.stderr)
            failed = True
            continue
        old = dict(walk(json.loads(path.read_text())))
        new = dict(walk(fn()))

        gone = sorted(set(old) - set(new))
        added = sorted(set(new) - set(old))
        changed = sorted(k for k in set(old) & set(new) if old[k] != new[k])

        if not (gone or added or changed):
            print(f"{name}: {len(new)} values, unchanged")
            continue

        failed = True
        print(f"\n{name}: {len(changed)} changed, {len(gone)} gone, {len(added)} new")
        for k in changed[:25]:
            print(f"  ~ {k}\n      was {old[k]!r}\n      now {new[k]!r}")
        for k in gone[:10]:
            print(f"  - {k}  (was {old[k]!r})")
        for k in added[:10]:
            print(f"  + {k}  (now {new[k]!r})")
        extra = len(changed) - 25
        if extra > 0:
            print(f"  … and {extra} more changed")
    return failed


if __name__ == "__main__":
    sys.path.insert(0, str(HERE))
    args = sys.argv[1:]
    mode = args[0] if args else "check"
    only = set(args[1:])

    if mode not in ("capture", "check"):
        sys.exit(f"usage: baseline.py [capture|check] [probe …]\n  got {mode!r}")
    if not reachable():
        sys.exit(f"nothing serving {URL} — start tools/preview_ui.py 8791 first")

    if mode == "capture":
        capture(only)
    else:
        sys.exit(1 if check(only) else 0)
