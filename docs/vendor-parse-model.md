# The parse model is a fork, and this is the record of it

CLAUDE.md's rule is *depend on maintained upstreams over owned forks*, and it
adds the condition under which breaking it is survivable: **record the source
revision and every local change, so a sync is a diff rather than an archaeology
exercise.** `forge/` was the last thing that earned this and it is gone. This is
the next one, written at the moment the dependency is taken rather than the
moment it is regretted.

## What is forked

| | Repo | Revision | License |
|---|---|---|---|
| **In use** | `huihui-ai/Huihui-Qwen3-4B-Instruct-2507-abliterated` | `c9bd464550d4078c72af0dd22aa18d0437868ce3` | apache-2.0 |
| **Its base** | `Qwen/Qwen3-4B-Instruct-2507` | `cdbee75f17c01a7cc42f958dc650907174af0554` | apache-2.0 |
| **The encoder it writes for** | `Qwen/Qwen3-VL-4B-Instruct` | `ebb281ec70b05090aa6165b016eac8ec08e71b17` | apache-2.0 |

Pin the revision, not the branch. A fork with 2,025 downloads against its base's
3.3 million is one person's upload, and it can be amended or withdrawn without
anyone noticing.

## The one local change

**Abliteration**, applied to the text weights only. The procedure identifies the
direction in activation space that produces a refusal and orthogonalises it out
of the weights — no retraining, no fine-tune, no new data. Everything else is
the base model, which is why capability loss is small: the sibling
`n0ctyx/Qwen3-4B-Instruct-Uncensored` reports **KL divergence 0.0785** against
its base with ~81% of previously-refused prompts answered, which is the order of
magnitude to expect here.

Neither card reports a capability benchmark. `tools/smoke_parse.py` is the
substitute and it is the reason it exists.

## Why the fork is justified here, when it usually is not

**The alternative is not "use the unforked model." It is "ship a tool that
refuses the user's material."**

`CAPTION_MODELS` already documents this failure at the layer below: a stock
instruct model declines on real people often enough to matter, and the decline
is *fluent* — it passes every check downstream and lands in a `.txt` sidecar.
The parse is worse in one specific way. With a schema bound, a refusal cannot
arrive as recognisable prose; it arrives as an **evasive storyline that
satisfies the schema**. Nothing downstream can tell it from a real one.

A storyteller's material makes this routine rather than rare. Violence,
intimacy, real people, a recruit on a latrine floor before he kills his drill
instructor. Those are the scenes the product exists for.

## The staleness already present, and what to check

The fork was cut **2025-08-07**. Its base repo has been modified since,
**2025-09-17**. A HuggingFace `lastModified` moves for a card edit as readily as
for weights, so this is a thing to check rather than a known divergence —
compare the base's file list and `config.json` at the two revisions before
assuming either way.

This is the ordinary condition of an unmaintained fork of a maintained upstream,
which is the worst of the three available positions and is accepted here with
the escape hatch below rather than pretended away.

## Replacing it, which is the point of this file

Abliteration is a **procedure, not a model**. It is deterministic given the same
calibration prompts and re-runnable against whatever Qwen ships next, so a stale
fork is an afternoon rather than a rewrite:

1. Take the new base at a pinned revision.
2. Run a harmful and a harmless prompt set through it, capturing residual-stream
   activations at each layer.
3. The refusal direction is the difference of the means. Pick the layer where it
   separates most cleanly.
4. Orthogonalise that direction out of the weights that write to the residual
   stream.
5. Score with `tools/smoke_parse.py` — **fidelity and compliance both**, against
   the same corpus, so the new artifact has a number comparable to the old one.

Step 5 is the part that is not optional. Abliteration can damage a model in ways
that never surface as a refusal and do surface as worse judgment, and judgment
is the entire job here — mental versus physical relations, inherent versus
contingent, phrasing a child so it continues its parent.

## What would retire this file

An official instruct model whose refusal behaviour does not break on a
filmmaker's material, at a size this job can afford. That is the maintained
upstream the rule prefers, and taking it would mean deleting this file rather
than updating it.
