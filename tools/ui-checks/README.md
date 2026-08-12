# UI checks

Playwright scripts that drive `tools/preview_ui.py` and assert what the console
actually does, rather than what a stylesheet says it should. They live here
because the alternative — a session scratchpad — is where the last several
rounds of them died, and each one encodes a fault that was expensive to find.

    /opt/homebrew/bin/python3.11 tools/preview_ui.py 8791
    python3 tools/ui-checks/check_neg.py

- `check_neg.py` — the negative-prompt toggle and the auto-growing prompt field.
  Covers the models that read no negative, the CFG that wakes it, and the cap
  the field stops growing at.
- `check_drop.py` — every drop target, asserting each one *cancels* dragover.
  A target that does not cancel never receives the drop at all, which is how the
  reference tray shipped dead.
- `measure_console.py` — console height against the 30% budget at three
  viewports.

They assert against `preview_ui.py`'s stubs, so they need no Modal account and
no GPU.
