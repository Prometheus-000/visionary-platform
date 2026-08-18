# The parse model is a fork, and this is the record of it

CLAUDE.md's rule is *depend on maintained upstreams over owned forks*, and it
adds the condition under which breaking it is survivable: **record the source
revision and every local change, so a sync is a diff rather than an archaeology
exercise.** `forge/` was the last thing that earned this and it is gone. This is
the next one, written at the moment the dependency is taken rather than the
moment it is regretted.

## It is wired, and there is no fallback behind it

As of this sprint the parse runs on these weights and only these weights. The
hosted path is **removed rather than deprecated**: `PARSE_MODEL`,
`_anthropic_key()`, the `anthropic_key` branches on `/api/token` and
`/api/state`, the `anthropic==0.42.0` dependency in `web_image`, and
`smoke_parse.py --backend hosted` are all gone. One interpreter, not two.

That is what pays for the semantic layer's "no new controls": a hosted model
needed an API-key field in Settings, and local weights need nothing at all,
because they are served off the existing `hf_cache` volume the captioner already
uses. `modal deploy app.py` stays the entire install.

It also means **this file is now on the critical path.** A fork that is merely
referenced can go stale quietly; a fork the parse cannot run without goes stale
loudly, and the replacement procedure below is the thing that makes that
survivable rather than alarming.

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

## The justification below was tested and did not hold

**Measured 2026-08-17 on live weights, ten charged fragments** — violence, real
people, an execution, intimacy, minors in jeopardy — **the unmodified base
refused none of them** (9/10 on-subject; the one miss is a dropped word, not
evasion) and beat this fork on preservation, relations, idempotency and
compliance at identical VRAM. The abliterated fork scored 10/10 on the same
corpus, so the fork is not *worse* at the thing it was cut for; there is simply
no gap for it to close.

That does not automatically retire the file — one corpus is one corpus, and a
refusal that appears on the eleventh scene is still the failure described below.
But the claim in this section is now a hypothesis with evidence against it
rather than a settled reason to carry an unmaintained fork of a maintained
upstream, which this file itself calls the worst of the three available
positions. **`Qwen/Qwen3-4B-Instruct-2507` at a pinned revision is the thing to
try first**, and `smoke_parse.py --refusal` is the corpus that would have to
find a failure for this file to keep its subject.

One thing that is *not* an argument for either model: on the enrichment runs the
base authored 12 of 18 fragments against the fork's 9, but it also mislabelled
provenance more often. Both are properties of `PARSE_RULES` and of asking the
model for `origin` rather than computing it — see CLAUDE.md — so neither belongs
in this comparison.

## The original justification, kept for the record

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
   `tools/stress_parse.py` drives it in a throwaway Sandbox, so a candidate that
   scores badly costs a few minutes of L4 and leaves no trace.
6. ~~Re-run `smoke_parse.py --sweep` against the served endpoint.~~ **Do not.**
   That was done, and both bounds turned out to be unrepairable in kind rather
   than merely mis-set: a share of characters is dominated by how much the
   person typed, so on a fragment a good enrichment scores 94% invention and an
   evasive document 93%. Run `--enrich --scenes` instead and read the four
   criteria — core subject, emotional tone, spatial logic, literal fidelity.
   See CLAUDE.md, "Prompt replacement, and what it cost to get there".

Step 5 is the part that is not optional. Abliteration can damage a model in ways
that never surface as a refusal and do surface as worse judgment, and judgment
is the entire job here — mental versus physical relations, inherent versus
contingent, phrasing a child so it continues its parent.

## What would retire this file

An official instruct model whose refusal behaviour does not break on a
filmmaker's material, at a size this job can afford. That is the maintained
upstream the rule prefers, and taking it would mean deleting this file rather
than updating it.
