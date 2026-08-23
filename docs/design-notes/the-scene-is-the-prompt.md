# The scene is the prompt

The prompt box is the last place in this app where a documented grammar is
presented as free prose. This note is the design that replaces it, and the
argument for why the replacement is a *structure* rather than a better textarea.

## What the H3 repo settled, and what it did not

`MiniMax-H3/skills/h3-prompt-writing` ships in the model repo: a SKILL.md, two
reference guides (`base-en.txt`, 15.8k; `ref-en.txt`, 23.6k) and a lockfile
pinning them by hash. It is the same move Krea made with `docs/expansion.txt` and
the same move this file already took on `KREA_EXPANSION`: **the people who
trained the encoder wrote the instructions for talking to it.**

So the *rewriter* is solved upstream. What it does not solve — and what reading
it makes unavoidable — is that the grammar it documents is far larger than what
`_compile_h3_prompt` emits. Every one of these is in the guide and in none of our
output:

| the guide asks for | we emit |
|---|---|
| `[Shot 1] … [Shot 2] …` segmenting the body | one undifferentiated blob |
| `[Shot N] At 00:03.500, the camera cuts to …`, strictly increasing, no timestamp on Shot 1 | no cut times at all |
| stable speaker IDs `(S1)`, `(S2)`, compound `(S1,S2)`, held across shots | none — dialogue is a clip-level pill |
| `<scenetrans>` when a line crosses a cut, `<cutoff>` when the clip truncates it | neither, because we have no cuts |
| style declared at the head of Shot 1 (`Live-action, cinematic, …`) | wherever the person happened to type it |
| camera motion as a natural action *inside* the shot, carrying type + amplitude + speed | one clause stapled at slot 40 |
| `retention_analysis` recording **where** each subject appears | a generic "must retain its facial structure" |

Read the list again and notice what kind of facts they are. Not one is a
sentence somebody could be expected to write. Every one is a **structural fact
the interface already knows or could know**: a cut time is a boundary between two
rows, a speaker ID is a cast member who has a line, a retention sentence is the
set of shots a name appears in. `_compile_h3_prompt` cannot emit them because the
composer never collected them — there is one text field, and a text field has no
shots in it.

That is the whole case. **The prompt box is not a bad control because it is
unstructured; it is a bad control because the structure exists and it is the one
surface that refuses to hold it.**

## The shape: a cast and a timeline

Two zones replace the field. Neither is a panel of settings — both are the
thing itself.

**The cast** is who and what is in the piece. A bucket per entity, and an entity
is a `character`, a `place` or a `thing`. Each bucket carries a name, and the
name is its `@handle` — the only text anyone types about it.

**The timeline** is what happens, one row per shot. A row is a sentence with the
cast in it as tokens: typing `@` opens the cast, picking one drops a handle. A
row also carries its own pills, its own beat width, and a keyframe if it has one.

Every field in the six-field document is then derived rather than asked for:

    subject_definitions   ← cast order, and what each bucket's slots were filled with
    summary               ← the first shot's line, cast resolved to nouns
    retention_analysis    ← for each subject, the shots its handle appears in
    detailed_description  ← the rows, in order, with cut times from the beats
    overall_soundscape    ← sound pills, aggregated across shots
    non_diegetic_music    ← the clip's score pill, or N/A

`retention_analysis` is the one to watch, because it is the field that goes from
a guess to a fact. Today it says `<Subject 1> must retain its facial structure,
hair and build.` With a timeline it says `<Subject 1> appears in [Shot 1] and
[Shot 3] and must retain its facial structure, hair and build.` The guide asks
for exactly that sentence and there has never been anywhere in this app that
knew it.

## Zone-to-role: a drop is a categorisation

A reference picture's *role* is the fact that makes "do not describe the picture
you attached" enforceable rather than advice — `SHOT_REF_ROLES` already says so.
What it does not do is make setting the role free: today a chip is attached and
then tagged, and `refs: string[]` runs alongside `refRoles: string[]` as two
positional arrays that have to be kept in step by hand.

So the role is decided by **where the file lands**. A character bucket has
slots — Face, Wardrobe, Body, Voice, Motion — and a place has Establishing and
Style. Dropping a photograph on Face is the tagging. There is no second gesture
and no menu, which is the same reason a region's LoRA is a dropdown on the box
rather than a caret-targeted token.

The slot table is also the file validation, declared once rather than branched:

    face wardrobe body establishing style object   image
    voice                                          audio
    motion                                         video

A file over a slot that cannot take it does not highlight. The rejection is the
absence of the invitation, which is cheaper than a toast and cannot be missed
after the fact.

## The pool is flat, and buckets hold pointers

One `Record<fileId, PoolFile>` for every file that has been dropped, keyed by
**content hash**. A bucket holds `{ fileId, slots: Slot[] }`.

Two things fall out and both are bugs the current arrangement has:

- **The same photograph in two buckets is one upload.** Two characters shot on
  the same day from the same still, or one picture that is both the wardrobe and
  the body reference, cost one entry and one encode. Keyed by hash rather than
  by name, because a file dragged from two folders is the same picture with two
  paths.
- **`<Picture N>` becomes derivable.** Numbering walks the cast in order and
  collects distinct file ids, so removing a bucket renumbers correctly and the
  two-parallel-arrays invariant is gone — not fixed, *unrepresentable*.

## The mention is text, and that is deliberate

The row is a `<textarea>` with a mirror over it, the same construction
`Field.tsx` and `marks.ts` already use, and for the reasons already recorded
there: it keeps the caret, the selection, the native undo stack and every chord
in `keys()`. A contenteditable buys chips and loses all four.

So a mention is stored as the literal text `@ava` and painted as a chip by the
mirror. That has a consequence worth stating rather than discovering: **edit the
handle and it stops being a mention.** The words turn plain and the shot no
longer claims that subject. That is not a failure mode, it is `remap`'s rule one
layer up — the user's text is the record, and everything derived from it is a
receipt. Renaming a cast member rewrites the handle across every row, as a
visible find-and-replace, because the alternative is storing `@{id}` and showing
the user a string they cannot safely edit.

Handles are therefore `[a-z0-9_]+`, lowercased off the name, and unique.

## Degrading to nothing

The rule `_compile_h3_prompt` already keeps, extended: **one shot, no cast, no
pills, and the compiled output is the typed text byte-for-byte.** A person who
opens this and types a sentence into the first row gets exactly what the prompt
box gave them. The structure appears as they use it and never before.

## Checking the output

The document is a disclosure under the timeline, showing the six fields as
`/api/compile` returns them — the same compiler the run uses, on the same CPU
container, because a preview with its own implementation is a preview that can
disagree with the run.

Each field is editable, and editing one **pins** it: that field stops
recomputing, the rest keep tracking the scene, and the pin is visible and one
gesture from gone. This is the escape hatch the four region coordinates are — the
numbers are the parameter and the rectangle is the primary, and dragging teaches
the numbers while the numbers never taught the dragging.

What it must not become is the place the work happens. If somebody is typing
into `detailed_description` to get the take they want, the timeline above it has
failed and no amount of polish on the sheet is the fix.

## What this vetoes

- **No shot inspector.** A shot's properties are on the shot's row. The moment
  there is a panel that shows "the selected shot", this is a node graph with
  better typography — the month-four failure Phase 6 names.
- **No asset manager.** The pool is not browsable. It is the set of files you
  have dropped, visible only as the thumbnails sitting in slots.
- **No template gallery.** The eight skills in the H3 repo are generators for
  ad, explainer and music-video shapes. They are a marketplace of somebody
  else's method, and the standing veto on that is already written.
