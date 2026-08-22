# LoRAs are not text

They were written in the prompt for one reason — so you could type a strength —
and everything else about that decision was the price. Chips with a value carry
the strength better, so the reason is gone and the price should go with it.
Design note and spec, 2026-08-21.

## The two things that were conflated

`CLAUDE.md` opens its LoRA section with *"LoRAs are written in the prompt, not
stacked under it"* and defends it well. The defence holds for one of the two
things that ended up in that field and not for the other, and separating them is
the whole of this change:

- **A trigger phrase is text.** It reaches the encoder. It has to be in the
  prompt, positioned where the sentence wants it, and it is words like any other
  words.
- **A LoRA is a file plugged into a module.** It never reaches the encoder at
  all — `stripLoras` deletes it from the string before `/api/generate` is
  called, and `readLoras` parses it into a `{path, unet, text_encoder}` stack
  travelling in its own field. It was only ever inline so a number could be
  typed beside it.

`+ LoRA` writes both today, in one gesture, which is why they read as one thing.
They are not.

## What the original decision actually bought, checked

Worth doing before reversing it, because two of the three claims survive and one
does not.

> *"A row per LoRA cost 56px plus a wrapped select — 380px of canvas for four
> filenames and eight digits."*

**True, and this does not bring rows back.** The chips collapse behind a
disclosure that costs nothing at rest.

> *"…and it still could not say the thing that matters most, which is **where in
> the sentence** the LoRA applies."*

**True in a region, false in the main prompt**, and the file says so itself, in
`useDocument.ts:196`:

> *"Put back at the end, because a token's position in the main prompt means
> nothing to the backend, which reads them into a stack."*

So the main prompt has been paying for inline syntax to buy a property only
regions have. In a region the position does mean something — but what it means
is *which box*, and a control living **on** that box says it without syntax.

> *"…in the prompt a fifth LoRA costs the canvas nothing."*

**True, and the disclosure matches it.** Four chips and one chip cost the same
one line.

## The rule

> **The composer is per-kind. Only the canvas and the gallery are shared.**

This reverses *"Image and video are one workspace. Shared prompt, canvas and
gallery; the switch is a chip inside the prompt field and the prompt survives
it."* The reversal is narrower than it reads: the canvas and the gallery *are*
still shared, and they are the two surfaces that made "one workspace" true. What
stops being shared is the composer, because everything in it is per-architecture.

The fault it fixes is live and silent today. Type `<lora:k3nan:1>` on the image
side, switch to video, and `readVidLoras` resolves it and loads a **Krea 2** LoRA
into a **Wan** run. Nothing says so — `loraNote` warns about names that resolve to
nothing and about models that read no LoRAs at all, and a LoRA for the wrong
architecture is neither.

## State

`prompt`, `negative`, `negOn`, `shot` and the new `loras` stay at the store root
as **the live buffer**. `setKind` swaps them with a dormant copy:

```ts
type Composed = { prompt: string; negative: string; negOn: boolean
                  shot: ShotPill[]; loras: LoraChip[] }

stash: { image: Composed | null; video: Composed | null }
```

**One live set, one dormant, and the swap is the only place both exist.** Moving
all five onto `ImageComposer`/`VideoComposer` is the more literal reading of the
rule and touches roughly thirty call sites, because `s.prompt` is read almost
everywhere. This touches `setKind`. It keeps the invalid state unreachable by the
same means — there is exactly one live set, so no component can read the wrong
one — which is `docFor`'s trick rather than a shortcut around it.

Two riders:

- **`docUndo` is cleared on switch, not stashed.** An undo slot that restores the
  other kind's sentence is worse than no undo.
- **`motion` moves into `vid`.** It is video-only already; it was at the root
  because the prompt was.

Both kinds keep their buffer. Switch away and back and the sentence, the pills
and the chips are exactly as they were — the same promise `img`/`vid` already
make about model, size, steps and seed.

## The chip

```ts
type LoraChip = {
  path: string                              // the identity /api/generate takes
  rel: string                               // what the chip reads
  strength: number                          // the circle
  textEncoder: number | null                // image, disclosed
  expert: 'high' | 'low' | 'both' | null    // video, disclosed; null derives
}
```

Name, then a circle carrying the value. Clicking the chip discloses the second
value — the text-encoder weight on the image side, the Wan expert on the video
side — because both are omitted far more often than not and `vidExpert`'s
filename derivation is right nearly always. `vidExpert` survives for exactly that
default; `expert: null` means "read it off the name."

**The circle carries `data-step="0.05"`.** The file already records why: fields
whose useful range is 1.0 to 1.4 need their own step, because *"a shift of 1.15
stepped by 8 leaves behind every value the model accepts."*

## The box

A **disclosure in flow**, built as one more `#shot-peek` rather than as a new kind
of thing:

```
▸ 4 LoRAs
──────────────────────────────────
  k3nan (1.0) ×    alxcn (0.8) ×
```

Four properties that pattern already has and a popover does not:

- **Zero pixels at rest.** `if (!loras.length) return null`, the rule
  `#shot-rail:empty` encodes — a row is affordable when it carries content and
  never when it carries one control.
- **In flow.** *Nothing sits on top of a render*: a popover floats over the
  canvas, this pushes the console, and `fieldMax()` already absorbs exactly that
  for the peek's own 184px panel.
- **It does not close on scroll**, which is `CLAUDE.md`'s standing objection to
  `Popover` and disqualifying for a box you type numbers into.
- The caret, the `--wash-2` panel, the `--line` border and `--r-control` are the
  peek's, unchanged.

Collapsed, it says `4 LoRAs` — a **word, not a pip**. The regions button failed
precisely here: a count riding half-outside it *"read as an error pip rather than
'2 regions'."*

## The doors

`+ LoRA` stays exactly what it is: a picker in the strip. It does not become the
summary — the shot side already shows why that division is right, and this is the
same division:

| | adds | shows |
|---|---|---|
| shot | `Shot` door in the strip | the pill rail, and `▸ what the model reads` |
| LoRA | `+ LoRA` door in the strip | `▸ 4 LoRAs` |

The only difference is that pills are worth showing at rest and LoRAs are not,
which is why one collapses and the other does not.

**Picking adds a chip and writes nothing.** No token, no strength, and **no
trigger phrase**. The picker row still *shows* the known phrase as text, because
that is information rather than an edit — you type it where you want it.

**On H3 the door and the box are both absent**, not disabled. A control the model
will ignore is worse than one that is not there.

**A region gets its own dropdown, on the region's card** (`regions/Inspector.tsx`).
One LoRA per region is the node's shape, so it is a dropdown rather than an
add-many: it shows the region's current LoRA or None, with the value circle
beside it. This is what replaces caret-targeting — `+ LoRA` writing into
"whichever of the two fields you last had the caret in" was answering *where*
with a guess about where you were looking.

**Picking one already picked removes it.** The menu ticks what is in the box
today and a second click takes it out; that survives, and it is what makes the
tick legible as a state rather than as decoration.

## Drag and drop is deleted

`web/src/lora/drag.ts` goes, with its private MIME type and its wiring in
`App.tsx`, `ui/Menu.tsx`, `regions/RegionLayer.tsx`, `console/Field.tsx` and
`lora/LoraButton.tsx`, plus `.menu button.draggable` in `ui.css`.

It exists to answer *where* — *"a click writes at the caret and so inherits
wherever the caret was; a drag names its own target."* Once every target owns a
control, there is no *where* left to answer: the canvas chips are in the canvas
box, a region's chip is on that region. A gesture that exists to disambiguate
scope is dead weight when scope is structural.

It is a real loss on one axis and worth saying so: dropping a LoRA onto bare
canvas created a new box holding it, which was one gesture for what is now two.
That is the cost, it is small, and it buys back a subsystem.

## Trigger phrases: nothing writes them, nothing warns

The picker stops inserting the phrase, and `loraNote` stops mentioning it.

`CLAUDE.md` records why the warning existed and the reason has not gone away: a
LoRA bound to a phrase *"is near-invisible until the phrase is in the prompt —
the weight loads, the render changes a little, and it reads as a LoRA that did
nothing."* That failure is still real. This is a decision to manage it by hand
rather than be told about it, and it is recorded as that rather than as a
discovery that the warning was wrong.

**The carve-out needs no new channel.** `/api/state` already serves
`trigger_word` per LoRA entry — it is how the picker knows the phrase at all — so
an agent driving the platform, or a check, reads the fact from the state
directly. The fact stays; the warning goes.

## Legacy syntax is text

`<lora:…>` in a prompt is now words. Nothing parses it, converts it, absorbs it
or migrates it. The platform is not public, and code that exists to carry
yesterday's format across is code that is obsolete tomorrow.

**One existing call survives to stop that decision rendering badly.**
`stripLoras` stays on the **send** path (`useGenerate.ts:113`, `useVideo.ts:52`),
so syntax left in a reused prompt never reaches the encoder as the literal word
"lora". It is the only thing in the codebase that still knows the pattern exists,
it costs nothing, and it is not a migration — it removes text on the way out
rather than reinterpreting it in the box.

## What is deleted

From `web/src/lora/tokens.ts`: `LORA_RE`, `Token`, `parseLoras`, `resolveLora`,
`loraAlternatives`, `insertLora`, `removeLora`, `nudgeLora`, `loraSyntax`,
`readLoras`, `readVidLoras`. Surviving: `loraIndex`, `LoraFile`, `loraNum`,
`vidExpert` and `stripLoras`.

From `web/src/lora/note.ts`: the trigger-phrase block, `No LoRA named "x"`, and
the ambiguity block. A chip is picked from a list, so it always resolves — those
three cases cease to exist. What survives: the stack cap, the model-reads-no-
LoRAs note, the same-LoRA-in-prompt-and-box note (now: in the canvas box and in a
region), and the region-takes-one note.

From `web/src/console/Field.tsx`: the `nudgeLora` keybinding. `caret.ts` stays —
`moveClause` and the region fields still register a sink through `caretProps`.

`tools/ui-checks/probe_lora.py` loses the rows asserting the three deleted notes
and gains rows for the chips.

## Testing

`tools/ui-checks/check_loras.py`, new:

- Picking two LoRAs puts two chips in the box and `▸ 2 LoRAs` on the summary.
- With none, the box is absent entirely — no row, no caret, nothing.
- The strength circle steps by 0.05 on ↑/↓, not by 1.
- The second value is hidden until the chip is opened.
- `/api/generate` receives the stack — path and unet per chip — and the prompt it
  receives contains no `<lora:`.
- A region's dropdown takes one and replacing it replaces rather than appends.
- On an H3 model neither the door nor the box is rendered.
- **The switch:** write an image prompt with two chips and a pill, switch to
  video, write a different prompt with different chips, switch back, and find
  both composers exactly as left.

`preview_ui.py` needs no new stub — `/api/state` already serves the LoRA index
that the picker reads.

## What this reverses in CLAUDE.md

Three passages, rewritten as claim-then-what-retired-it rather than deleted:

1. **"LoRAs are written in the prompt, not stacked under it."** Retired for the
   main prompt by the position argument above; the row-per-LoRA objection it was
   really written against is answered by the disclosure rather than reopened.
2. **"Image and video are one workspace. Shared prompt…"** Narrowed: the canvas
   and gallery are still shared, the composer is not.
3. **The trigger-phrase note.** Kept as a description of a real failure, with the
   decision not to warn about it recorded beside it.


## A note on sequencing

This is one coherent idea and it may be two plans. The chips and the drag
deletion are self-contained and testable on their own; the per-kind composer
touches `setKind` and every surface keyed to the prompt. If the plan comes out
unwieldy, the split is **chips first, then the switch** — chips work fine against
a shared composer, and the switch work is what makes the Krea-2-LoRA-into-Wan
fault unrepresentable rather than merely unlikely. That ordering never leaves a
broken intermediate.
