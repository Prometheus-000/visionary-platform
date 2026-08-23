"""
Can a Stop press be lost? Enumerate the interleavings rather than race them.

    python3 tools/smoke_stop.py

**The bug this exists for.** The stop flag was a field on the job record, and
every `_publish` rewrites that record with get-update-put against a *network*
Dict. `_publish` takes `_PUBLISH_LOCK` — which is process-local, and the Stop
route runs in the web container, a different process, so it never took that
lock. The GPU container's publish was already holding a copy read before the
press and put it back with `stop: False`.

On the generate path a publish lands on every tqdm line, several times a second,
so the window was open more or less continuously. The symptom was a run that
would not stop and got killed from the Modal dashboard instead.

The fix is a key nobody merges — `stop:{job_id}`, written only by the Stop route
and read only by the job. A merge cannot clobber what it does not touch.

**Enumerated, not raced.** Two actors, four operations, and every order they can
happen in. A test that sleeps and hopes reproduces the bug on the runs where it
would not have mattered and passes on the ones where it would.
"""

from itertools import permutations

STORE_KEY = "j"


def interleavings():
    """Every order of the publisher's get/put and the presser's get/put that
    keeps each actor's own two operations in sequence."""
    ops = ["pub_get", "pub_put", "press_get", "press_put"]
    for order in set(permutations(ops)):
        if order.index("pub_get") < order.index("pub_put") \
                and order.index("press_get") < order.index("press_put"):
            yield order


def replay(order, merged: bool) -> bool:
    """Run one interleaving. True if the job would see the stop."""
    store = {STORE_KEY: {"status": "running", "step": 5, "stop": False}}
    held = {}
    for op in order:
        if op == "pub_get":
            held["pub"] = dict(store[STORE_KEY])
        elif op == "pub_put":
            # The publisher's whole point: merge its own field and write back.
            store[STORE_KEY] = {**held["pub"], "step": 6}
        elif op == "press_get":
            held["press"] = dict(store[STORE_KEY])
        elif op == "press_put":
            if merged:
                store[STORE_KEY] = {**held["press"], "stop": True}
            else:
                # Its own key. The publisher never reads or writes this.
                store["stop:j"] = True
    return bool(store.get("stop:j") or store[STORE_KEY].get("stop"))


def main() -> int:
    orders = sorted(interleavings())
    fails = []
    for merged, label in ((True, "stop as a merged record field"),
                          (False, "stop as its own key")):
        kept = [o for o in orders if replay(o, merged)]
        lost = [o for o in orders if not replay(o, merged)]
        ok = (len(lost) > 0) if merged else (len(lost) == 0)
        print(f"  {'ok  ' if ok else 'FAIL'}  {label}: "
              f"honoured {len(kept)}/{len(orders)} interleavings")
        if merged and lost:
            for o in lost:
                print(f"          lost when: {' -> '.join(o)}")
        if not ok:
            fails.append(label)
    print()
    if fails:
        print("  " + "\n  ".join(f"unexpected: {f}" for f in fails))
        return 1
    print("A merged field loses the press on any order where a publish\n"
          "straddles it. A key of its own survives every order there is.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
