"""
Does a long poll answer on change and on the deadline, and never before either?

    python3 tools/smoke_longpoll.py

**What this protects.** `/jobs/{id}` on the job API holds a request open until
the job record differs from the token the client already has, or `wait`
seconds pass. Two ways for that to be wrong, and neither shows up as an error:
answering *early* while nothing has changed is the page's 400 ms poll again with
a longer name, and answering *late* — or never — when the record has changed is
a client that learns of a finished render at the deadline rather than at the
change. A third: a token that differs for an identical record makes every poll
return at once, which is the early case wearing a fingerprint.

Driven at a fake clock, with a fake Dict, so a thirty-second wait costs nothing
and the assertions are about ticks rather than wall time.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _from_app import pull  # noqa: E402

G = pull({"_await_change", "_job_token", "JOB_WAIT_MAX_S", "JOB_WAIT_TICK_S"})
await_change, token_of = G["_await_change"], G["_job_token"]
TICK = G["JOB_WAIT_TICK_S"]


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0
        self.slept: list[float] = []

    def __call__(self) -> float:
        return self.now

    def sleep(self, s: float) -> None:
        self.slept.append(s)
        self.now += s


def run(store: dict, since: str, wait: float, *, change_after: int | None = None,
        changed: dict | None = None):
    """Poll `store` from a fake clock; after `change_after` reads, swap in `changed`."""
    clock = FakeClock()
    reads = {"n": 0}

    def read():
        reads["n"] += 1
        if change_after is not None and reads["n"] > change_after:
            store.clear()
            store.update(changed or {})
        return dict(store) if store else None

    rec, token = await_change(read, since, wait, clock=clock, sleep=clock.sleep)
    return rec, token, clock, reads["n"]


failures = 0


def check(name: str, ok: bool, detail: str = "") -> None:
    global failures
    print(f"{'ok  ' if ok else 'FAIL'} {name}" + (f" — {detail}" if detail else ""))
    failures += 0 if ok else 1


running = {"status": "running", "phase": "generate", "step": 3, "total_steps": 8}
done = {"status": "completed", "files": ["gen1_00.png"]}

# 1. The token is a function of the record and nothing else.
check("identical records share a token", token_of(dict(running)) == token_of(dict(running)))
check("key order does not change the token",
      token_of({"a": 1, "b": 2}) == token_of({"b": 2, "a": 1}))
check("a changed field changes the token",
      token_of(running) != token_of({**running, "step": 4}))

# 2. A client with no token gets the current record at once.
rec, tok, clock, n = run(dict(running), since="", wait=30.0)
check("first ask answers immediately", n == 1 and not clock.slept and rec == running)
check("the reply carries the record's token", tok == token_of(running))

# 3. Nothing changes: the reply comes at the deadline, not before, and not after.
rec, tok, clock, n = run(dict(running), since=token_of(running), wait=3.0)
check("unchanged record waits out the deadline", clock.now - 1000.0 == 3.0,
      f"returned at +{clock.now - 1000.0:.2f}s")
check("it polls once a tick while waiting", n == int(3.0 / TICK) + 1,
      f"{n} reads for a 3 s wait at {TICK} s ticks")
check("the unchanged reply keeps the same token", tok == token_of(running))

# 4. The record changes mid-wait: the reply comes within one tick of it.
rec, tok, clock, n = run(dict(running), since=token_of(running), wait=30.0,
                         change_after=4, changed=done)
check("a change answers within one tick", rec == done and clock.now - 1000.0 == 4 * TICK,
      f"returned at +{clock.now - 1000.0:.2f}s after 4 unchanged ticks")
check("the reply's token is the new record's", tok == token_of(done))

# 5. A job with no record yet reads as unknown, and the deadline still applies.
rec, tok, clock, n = run({}, since="", wait=30.0)
check("a missing record answers unknown at once", rec == {"status": "unknown"} and n == 1)
rec, tok, clock, n = run({}, since=token_of({"status": "unknown"}), wait=1.0)
check("waiting on unknown also times out", clock.now - 1000.0 == 1.0)

# 6. A zero or negative wait is one read.
rec, tok, clock, n = run(dict(running), since=token_of(running), wait=0.0)
check("wait=0 is a plain read", n == 1 and not clock.slept)
rec, tok, clock, n = run(dict(running), since=token_of(running), wait=-5.0)
check("a negative wait is a plain read", n == 1 and not clock.slept)

print()
if failures:
    sys.exit(f"{failures} check(s) failed")
print("all long-poll checks passed")
