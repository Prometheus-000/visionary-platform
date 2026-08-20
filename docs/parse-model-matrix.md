# The Parse Model Matrix — why the document layer failed, and what replaced it

The record of a route change, gathered into one place. It was previously spread
across CLAUDE.md ("Prompt replacement, and what it cost to get there") and six
commit messages. Nothing here overrides CLAUDE.md; it consolidates the finding
so the next reader does not have to reconstruct it from a diff.

---

## The finding, in one line

Every gate between the model's interpretation and a render was a **string
comparison against what you typed** — and the thing those comparisons were
standing in for is *did the model replace what I meant?* Those are different
questions, and on the input that matters they point in **opposite directions**.

---

## Why character matching was the wrong criterion

Four checks, all lexical:

| check | what it asked | where |
|---|---|---|
| `_preserved` | every run marked "derived" must appear verbatim in your prose | `app.py:7651` |
| coverage floor | ≥65% of your characters must be accounted for | `app.py:7505` (deleted) |
| `insertionOnly` | the compiled text's gaps must appear verbatim in what you typed | `web/src/console/marks.ts:150` |
| `_document_matches` | the compiled string must equal the box | (deleted) |

### It punished correct reading

On fragmentary, self-correcting input — which CLAUDE.md calls **the normal
case** — reading it right means dropping words:

```
night. no, late afternoon              → late afternoon · night     coverage 59% → REFUSED
she's angry. or maybe just tired       → she · angry · tired        coverage 48% → REFUSED
something like a courtroom but colder  → a courtroom · colder       coverage 31% → REFUSED
```

**Five of six silent fragments were refused for interpreting correctly.** The
corpus the threshold was swept over even assumed the model would keep "maybe"
and "some kind of" as text — while the rules tell it to tidy them away.

### And it was blind to the failure it existed to catch

This passes every check — **0% invention, 100% coverage**:

```
a rifle sits on a latrine floor at night / a recruit across his knees
```

Every character is yours. The rifle and the recruit have **swapped places**. A
one-word negation — `no rifle across his knees` — scores 3% and sails through.
So the criterion could not see **meaning inversion** at all, while refusing
correct interpretations constantly.

### The measured result

- The layer reached **0% of renders on finished prose**, **13% on fragments**.
- Blind-judged on the pictures, the document approach went **0 wins in 30**
  across two model sizes and two rule sets — always compressing, **median
  0.73×**, shorter than its input **10 times out of 10**.

---

## The two reframes that changed the route

**"I don't care if it preserves my words — I care whether the image exceeds my
expectations."** Everything being measured was *fidelity-to-input*. The actual
objective is **amplification of intent**. Under that objective, `invention 0.0%`
stops being restraint and becomes the **headline failure**.

**"It's not prompt enhancement, it's prompt replacement."** Enhancement is your
sentence with clauses added — which is what all four gates were built for.
Replacement is a **new prompt written for the encoder**, with your prose kept as
the record. The apparatus was not badly tuned for that; it was **precisely tuned
against it**: the prompts that produced the good pictures score **100% invention
against a 64% ceiling**.

---

## Why the current route

- **One operation, prose out, no document.** The element schema was the
  mechanism — a grammar whose unit is a short tagged fragment makes a model
  **decompose** where it needed to **write**. Rules rewrites never fixed it
  because the rules were not the constraint; the **return shape** was.
- **Krea's own `expansion.txt`, vendored verbatim.** The people who trained the
  encoder wrote the prompt for talking to it. Its rule 7 makes expansion
  conditional, which collapsed three buttons into one.
- **On Krea 2's own resident encoder.** Qwen3-VL-4B is already on the card;
  `lm_head` is stripped from the repackage but `tie_word_embeddings: true`, so
  the head is free. **The writer and the reader are the same model.**
- **Trust became structural rather than lexical.** You decide whether to run it,
  you see what came back, you edit or discard it. **Nothing reaches the encoder
  that wasn't on screen first** — which is what the marks were an expensive way
  of approximating.

---

## Result

**3 wins, 1 loss, 6 ties blind, against 0-for-30.** Modest, and the first thing
that ever beat doing nothing. All three wins are fragments the old validator
**refused** — the cases it threw away are exactly the cases the feature exists
for.

---

## What is still true, and where it is measured

- **Contradiction is enforced by nothing.** `empty diner, 3am` coming back with
  daylight is caught by no arithmetic over characters. What stands in for the
  missing check is that the replacement is **visible and editable** — a sentence
  you can read and delete. If that ever ends up behind a collapsed disclosure
  nobody opens, the trade stops being sound.
- **The measurement that counts is the picture, not the string.** Render the
  pair and judge the output against the original description:
  `tools/does_it_help.py`, `tools/prompt_ab.py`, `tools/judge_renders.py`.
  `tools/smoke_parse.py --enrich --scenes` is the parse evaluation, read against
  four criteria — core subject extraction, emotional tone transfer, spatial
  logic, literal feature fidelity — rather than a totalled lexical score.
- **What survives from the document layer, dead but on disk:** `useDocument`,
  `marks.ts`, `Reroll.tsx`, `insertionOnly`, and the `modules` plumbing.
  `_document_trust` / `_trusted_modules` (`app.py:7716`, `app.py:7783`) remain on
  the compile path as a structural zero-check, not a threshold. Kept so the
  measurement can be re-run and the deletion is one commit somebody makes on
  purpose.

> **You cannot measure intent with a diff.** The gate that refuses a correct
> reading and passes an inverted one is not mis-tuned — it is measuring the
> wrong quantity. Judge the render, and make the interface the trust surface.
