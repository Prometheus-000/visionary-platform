"""
An independent grader for H3 documents — **theirs, not ours.**

Source: https://huggingface.co/spaces/geocine/MiniMax-H3-Prompt-Enhancer-2.6B
Revision: abf87a8b0fbd81f471122a39f639302af9814847
Files: minimax/{paths.py,formatting/*,scoring/*}, copied verbatim, nothing patched.

Vendored, which this codebase treats as a last resort — so here is the case.
It is **test-only**: nothing on a render path imports it, and deleting it costs
one section of `smoke_scene.py` and no behaviour.

What it buys is the one thing our own tests structurally cannot. `smoke_scene.py`
asserts the rules *we read out of the guides*, so it cannot catch us having
misread them — it would assert the misreading just as confidently. This is a
second reading of the same documents by somebody who then trained a model on
the corpus, which makes disagreement between the two a real signal rather than
an argument.

Reimplementing its checks would defeat the point exactly the way a hand-written
copy of `SHOT_VOCAB` would: a grader we wrote agreeing with a compiler we wrote
is one opinion, twice.

To refresh: re-download at a newer revision and update the line above. There is
nothing local to merge.
"""
