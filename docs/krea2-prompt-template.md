# The Krea 2 prompt template

> **This is the encoder's model, not the user's — and that distinction is the
> whole point of it.** Everything below describes how Qwen3-VL reads a prompt, so
> the compiler needs it and the interface must never expose it. A user's model is
> whatever they invent: their own words, their own groupings, their own
> `key:value`. Promoting this taxonomy to the surface — chips labelled
> `DECLARATION` and `BEHAVIOUR`, an arsenal filed into these categories — is
> making the pipeline's data model into the user's worldview, which is the one
> mistake that cannot be corrected later without breaking everything built on it.
> Chips carry the user's words. This table stays underneath.

## Context

Krea 2 reads its prompt through **Qwen3-VL-4B-Instruct** ([app.py:565][te]).
`app.py:936` already records why that matters — it parses grammar, so the binding
between an adjective and the noun it modifies is resolved by sentence
construction rather than guessed. What that note stops short of saying is what
eleven known-good generations demonstrate:

**the prompts that adhere are the ones shaped like a caption Qwen3-VL would have
written of the target image.**

That is the whole finding. The encoder is a captioner's sibling, so a prompt in
the grammar of a caption is a prompt in its native distribution. Everything below
is that principle made fillable — a set of variables, and the rules that turn
them into the sentence.

The evidence is three Gucci campaign recreations, where the model never saw the
original and worked from the prompt alone, and eight prompt/output pairs
published with the model. They are in `Prompt to generation examples/`.

[te]: ../app.py

---

## The governing rule: relate everything, or it becomes an entity

Everything else in this document is an instance of one finding.

**Anything described as a standalone unit becomes a standalone thing in the
picture.** Independence *is* entity-hood, to this model. A subject with no stated
tie lands in his own sector of the frame. Light described in its own clause
becomes a visual event rather than a property of surfaces. A compositional
property promoted to the opening slot becomes a character. Those are not three
faults; they are one fault at three scales.

### An element either hangs off an anchor or becomes one

There is no third state, and that is the whole mechanism.

The **anchor** is the token a clause opens with and everything after it attaches
to — the same anchor the subject rule is built on. Anything that hangs off one is
a *property* of that thing. Anything that hangs off nothing **becomes** an anchor,
which is to say a thing in its own right. `hard side light` has nothing above it,
so light becomes an entity in the picture. `sunlight raking across her cheekbone`
hangs off *her* and stays a property of the subject. Unattached is never absent;
it is a position, and it is the loud one.

This gives the definition of a module that taxonomy could not:

> **A module is an anchor plus everything that hangs off it.**

Which settles where the boundaries fall, what belongs inside, and what has to
happen between — by mechanism rather than by category.

It also settles the syntax question at both levels. **Inventory inside an anchor
is correct**; the wardrobe list is a pure comma-separated inventory
— `a purple beret, a sheer black lace long-sleeved top, a wide studded black belt
and a teal suede skirt, with tall tan leather boots` — and it lands in every
example, because every item hangs off the woman. **Inventory between anchors is
the failure**, because peers with nothing above them are exactly a set of
entities. Same operation, opposite verdicts:

| | Order | Syntax |
|---|---|---|
| Between anchors | the author's choice | **relate** |
| Within an anchor | fixed by binding | **list** |

The compiler is currently correct at one level and wrong at the other by this
test. `wear[]` comma-joins inside a subject — atomic, right. `_shot_sentence()`
comma-joins *between* groups — organism, and that is the
`Lit by…, lit by…, in…, on…` pile.

Inventory is good at the atomic level and fatal at the organism level. You can
list what is inside a liver; you cannot understand a body by listing liver, heart
and lungs. A parts list of a body describes a corpse.

### Anchoring is a merge, and that is how a strip shows a relation

A linear arrangement cannot draw a graph, which looked for a while like a hole in
any strip-shaped surface. It is not one, because **anchoring is not an edge — it
is a merge.** An anchor plus its dependents is *one module*, so bound elements
move together for the reason that they are together. Coupling is the display.

Two gestures carry the entire relational model:

- **Merge** — drag light onto the subject and they become one element. The light
  is a property of them, its extent folds into theirs, and two competing warm
  spots become one.
- **Split** — drag it back out and watch it take the heat with it as it goes.

Which teaches the governing rule without a word of explanation: nobody has to be
told that an unbound element becomes a character once they have seen it happen
and seen it undone.

It also unifies three operations that look distinct and are not:

| Merging | Produces |
|---|---|
| two subjects | a collective — `a couple`, not `a woman next to a man` |
| light into a subject | light as a property of them rather than an entity |
| a prop into a subject | they are holding it |

`An element either hangs off an anchor or becomes one` — merged is *hangs off*,
separate is *is one*, and the strip has no third state either. That is what makes
it a faithful picture of the model rather than a diagram of it.

And merge is nesting, so it is the same gesture at every depth: a scene contains
shots, a shot contains modules, a collective contains subjects.

### The scope limit: inherent relations are already in the model

"Relate everything" taken literally produces nonsense — *her hands, attached to
her arms, attached to her body* — and every token spent on structure the model
already holds is attention taken from something that needed it.

**State contingent relations. Never state inherent ones.**

| Relation | Kind | Needs stating |
|---|---|---|
| hands → a person | inherent | no |
| leaves → a tree | inherent | no |
| light → skin | contingent | **yes** |
| one subject → another | contingent | **yes** |
| a handbag → a person | mixed | **the attachment point does** |

That last row explains a drift catalogued elsewhere in this document without an
explanation. `one hand holds a patterned handbag at her hip` returned a strap
over the shoulder: a handbag *near a person* is inherent enough that the model
supplies it, while *where it attaches* is wholly contingent and had to survive on
its own. The inherent half held and the contingent half is precisely what failed.

It also explains why extent spent on a part accrues to its whole. **The inherent
binding keeps pulling the part back** — a hand cannot be accidentally detached
from a person, however much is written about it. That is the model being correct,
not a limitation, and it means there are exactly two ways to make a part an
anchor, both of which work by removing the whole from contention:

- **Omit the whole.** `Weathered hands, knuckles scarred, resting on a table` —
  no person named, so the hands are the subject.
- **Frame so tightly that the part becomes the whole.** The macro eye does this:
  `An extreme close-up portrait featuring pale, freckled skin and a single blue
  eye` never really makes the person the subject.

**So bind every element to what it acts on.** Do not describe light — describe
light *on skin*, on terrain, on a wall, or in the shadows it throws. The adhering
examples do this without exception:

| Clause | Bound to |
|---|---|
| `harsh, direct lighting **highlights intricate skin pores** … **isolating the brightly lit features** against a pitch-black background` | skin, features, background |
| `golden hour sunlight **hitting rocky orange terrain and green vegetation**` | surfaces |
| `hard midday sun from the left **throwing short crisp shadows**` | its own effect |
| `**lit by a fluorescent tube and two white globe pendant lamps**` | fixtures in the room |

The one light clause in the set bound to nothing — `Even shadowless light with no
visible source`, which binds only to absences — belongs to the corridor, the
flattest of the three recreations.

**This subsumes the shadow rule stated further down.** "The light clause always
ends in shadow behaviour" is not about shadows being important; it is light bound
to its effect, and shadows are simply the commonest available binding. The
general rule is the binding, not the shadow.

**And it is why "AI slop" and "lacks intent" name the same thing.** An inventory
is precisely a set of isolated entities. Intent lives in relation, so a
description in which everything is an entity and nothing is a relation produces a
picture that is correct in every part and authored in none.

**Consequence for any interface built on this.** A surface that draws elements as
independent objects — a row of separate chips, a list of fields, a stack of tags —
is a picture of the failure mode. Extent and order are properties *of* elements
and a strip shows them well; relation is what makes the picture, and it has to be
visible too.

## The variables

Three fill modes per variable. This is the tri-state CLAUDE.md's Phase 6 calls
**derived or invented, always visible**, arriving early and in the composer.

- **`user`** — you supply it. The template never invents content.
- **`auto`** — the template supplies it, marked as invented and cheap to reroll.
- **`off`** — omitted unless asked for. Every low-reliability slot is here.

A slot marked in the **LoRA** column can be filled by picking a weight instead of
typing — its trigger word becomes the slot's value. See *LoRAs, by role* below.

The line between the modes: **the template fills craft, you fill the picture.**
Camera-department decisions have defaults that are right most of the time;
wardrobe and place do not, and a template that invents a garment is inventing the
shot.

### Frame — one per image

| Variable | Type | Mode | LoRA | Template supplies | Example |
|---|---|---|---|---|---|
| `register` | enum | `auto` | — | `photographic` | `photographic` · `illustrated` |
| `medium` | phrase | `off` | `style` | — | `1990s vintage anime style cel animation` |
| `count` | integer | `user` | — | — | `3` → emits `Three` |
| `kind` | noun | `user` | — | — | `figures` · `young people` |

`count` emits as a literal number word and is the first token in the prompt.

### Place — one per image

| Variable | Type | Mode | LoRA | Template supplies | Example |
|---|---|---|---|---|---|
| `place` | noun phrase | `user` | `location` | — | `a hotel corridor` |
| `verb` | verb | `auto` | — | from `place` | `standing` · `sitting` · `crowded into` |
| `surfaces[]` | 2–4 × `{material, colour, noun}` | `user` | — | — | `{glossy, pink, square tiles}` |
| `relation` | phrase, per surface | `off` | — | — | `above the tile line` |
| `depth_cue` | phrase, on the last surface | `auto` | — | from `place` | `running away from the camera` |

### Subject[] — ordered left to right

| Variable | Type | Mode | LoRA | Template supplies | Example |
|---|---|---|---|---|---|
| `position` | enum | `auto` | — | from clause count + index | `On the left` · `In the centre` |
| `anchor` | phrase | `user` | **`identity`** | — | `a red-haired young woman` · `k3nan` |
| `noun` | noun | `user` | — | — | `woman` · `small girls of about eight` |
| `verb` | verb | `auto` | — | from `place` | `stands` · `leans` · `sits` |
| `pose` | one clause | `off` | `action` | — | `her forearm against the wall` |
| `wear[]` | 1–5 × `{material, colour, noun}` | `user` | `wardrobe` | — | `teal suede skirt` |
| `pattern` | on exactly one `wear` entry | `off` | — | — | `embroidered with small birds` |
| `holds` | `{object, attachment}` | `off` | — | — | `a handbag at her hip` |
| `relates_to` | `{index, clause}` | `off` | — | — | **physical only** — `both arms wrapped around his waist` |
| `collapse` | `{indices, phrase}` | `off` | — | `dressed identically in` | merges subjects into one clause |

`wear[]` emits **head → torso → waist → legs → feet**, always, whatever order it
was filled in.

### Secondary[] — everything in frame that is not a subject

| Variable | Type | Mode | LoRA | Template supplies | Example |
|---|---|---|---|---|---|
| `noun` | noun phrase | `user` | `prop` | — | `A full-grown tiger` |
| `verb` | verb | `off` | — | — | `walks slowly` · `perched` |
| `frame_position` | phrase | `user` | — | — | `across the foreground from the left` |
| `pose` | clause | `off` | — | — | `with its head lowered` |
| `relation` | clause | `off` | — | — | `passing in front of the bench` |
| `crop` | clause | `off` | — | — | `partly cropped by the frame edge` |

`frame_position` is **frame-relative, never world-relative** — `in the blurred
right foreground`, `framing the top edge`, `perched on a rock to the left`, `the
left ear softly blurs out of focus`. This is the slot that carried the tiger, the
lilies, the monkey and the copper hair, and it is the only place in the template
where position is stated in terms of the picture rather than the room.

### Behaviour — one, covering every subject

| Variable | Type | Mode | Template supplies | Example |
|---|---|---|---|---|
| `gaze` | phrase | `auto` | `face the camera directly` | `look toward the camera` · `eyes lowered` |
| `expression` | adjective | `auto` | `expressionless` | `relaxed and unsmiling` |

### Light — one

| Variable | Type | Mode | Template supplies | Example |
|---|---|---|---|---|
| `quality` | phrase | `user` | — | `Hard midday sun` · `Even shadowless light` |
| `direction` | phrase | `auto` | `with no visible source` | `from the left` · `from above` |
| `shadows` | phrase | `auto` | from `quality` | `throwing short crisp shadows` |
| `background` | phrase | `off` | — | `against a pitch-black background` |

`shadows` is `auto` rather than `off` because it is the token that does the work:
a light clause that does not end in shadow behaviour is the one reliable way to
lose the lighting.

### Camera — one

| Variable | Type | Mode | Template supplies | Example |
|---|---|---|---|---|
| `format` | phrase | `auto` | `medium format` | `35mm` |
| `height_axis` | phrase | `auto` | `at eye level` | `straight on at chest height` |
| `composition` | phrase | `auto` | from clause count | `rigidly symmetrical` · `tightly framed` |
| `depth_of_field` | phrase | `auto` | `sharp from front to back` | `the whole room in focus` |

### Render — illustrated register only, tail

| Variable | Type | Mode | Template supplies | Example |
|---|---|---|---|---|
| `attributes[]` | phrases | `off` | — | `flat shading` · `grainy paper texture` |

---

## LoRAs, by role

A slot can be filled by typing, or by picking a LoRA and letting its trigger word
fill it. **The role vocabulary is `SHOT_REF_ROLES` (`app.py:5161`), unchanged** —
`identity`, `wardrobe`, `location`, `style`, `prop`, `action`. A reference picture
and a trained weight are two ways to supply the same role, so they share one
vocabulary rather than getting a parallel one.

| Role | Slot it fills | Trigger lands in |
|---|---|---|
| `identity` | `subject[n].anchor` | that subject's clause |
| `wardrobe` | `subject[n].wear[i]` | that garment phrase |
| `action` | `subject[n].pose` | that subject's pose clause |
| `style` | `medium` | the declaration |
| `location` | `place` | the place clause |
| `prop` | `secondary[n].noun` | that element's clause |

### The trigger is content; the token is a directive

These are two different things and they go to two different places.

- **The trigger word is slot-scoped**, because it is language and it binds by
  position. `k3nan` in a subject anchor is bound by the same mechanism as `a
  red-haired young woman` — it is the first token of its clause and everything
  after attaches to it. A wardrobe trigger in a `wear[]` entry binds to that
  garment. Put the same trigger in the tail and it binds to nothing, which is
  mechanism 7.
- **The `<lora:name:w>` token is collection-scoped**, because it is markup.
  `stripLoras()` (`web/src/lora/tokens.ts:149`) removes every token before the
  text reaches the encoder — "the tokens are markup for this page, not language"
  — so its position cannot affect binding. It gathers once, in one place, and the
  emitted prose stays clean.

### Why auto-filling the trigger is safe here and was not before

`removeLora()` carries the note (`tokens.ts:313`): the trigger *used* to be
inserted by the picker and stripped on removal, and that was taken out because
"a picker click that deletes a word out of your sentence is the worse surprise."

The objection is real and it is about **free text**, where there is no defined
place for a trigger to go and no way to know which words the picker owns. A slot
answers both: the trigger belongs to that slot's value, and deselecting the LoRA
clears it from that slot and nothing else. The template owns the text, so removal
is exact. That is the condition the earlier design could not meet.

### Where a masked LoRA goes instead

An `identity` LoRA on a subject that **has a region box** emits its token into
that box's field rather than the main prompt — one per box, which is the node's
shape and the whole point of regional masking. Same subject with no box, and the
token goes to the main prompt and applies to the canvas.

The trigger word does not move in either case. It stays in the subject's clause,
because that is where it binds.

### The budget

`MAX_LORAS = 6` (`app.py:2580`) across the prompt, and one per region box. Three
subjects carrying identity, wardrobe and action each is nine, so a template that
offers a LoRA on every slot can exceed the cap without anything looking wrong.
The count is worth stating before the run, in the shape the note under the prompt
field already uses — by name, on the CPU container, never a blocking error.

### Where a trigger word comes from

Three sources, and the third is the one that bites.

1. **First-party style LoRAs** — `KREA_STYLE_LORAS` (`app.py:851`) maps each of
   the nine names to its trigger, already surfaced as the catalogue entry's
   `note`. `darkbrush` → `monochrome ink wash style`.
2. **Anything trained here** — the training job carries `trigger_word`
   (`app.py:2359`), and the dataset records coverage against it.
3. **Anything pulled off Drive or migrated in** — no trigger is recorded
   anywhere, and there is no way to derive one from a filename.

Case 3 must **say so** rather than emit nothing. A weight that loads and does
nothing because its trigger was never typed is the failure this whole section
exists to prevent, and it is indistinguishable from a bad LoRA unless the page
names it.

## Where the values come from

Three sources fill one slot machine, and the split is by whether the vocabulary
is closed.

- **Typed** — `place`, `surfaces`, `anchor`, `wear`, `secondary`. Open
  vocabulary: this is the picture, and it cannot be a menu.
- **Pills** — framing, angle, light, tone. Closed vocabulary: this is the craft,
  and `SHOT_VOCAB` (`app.py:5207`) already holds it, with exact phrasings and
  animated tiles.
- **LoRAs** — trigger words into the slots marked in the LoRA column above.

### The shot palette is already the tail of this template

`SHOT_VOCAB`'s four image-side groups are the Light and Camera tables of this
document, built and shipped. The machinery matches part for part: `slot` is the
clause order, `join` is how clauses fold, `pick: one` / `pick: many` is `enum`
versus `[]`, and `image: bool` is which register reads it. There is nothing to
duplicate — the vocabulary grows a head, and the existing groups become its tail.

Today every image group sits at `-10`, so all four fold into one sentence ahead
of the prompt. The template's slot order separates them by what they modify:

| slot | group | source |
|---|---|---|
| −40 | declaration — `register`, `medium`, `count`, `kind` | new |
| −30 | place — `surfaces`, `depth_cue` | new |
| 0 | subjects | new — was the free textarea |
| 10 | behaviour | new |
| 20 | secondary | new |
| 30 | **light, tone** | existing, moved off `-10` |
| 40 | camera — **framing, angle**, format, height, DoF | existing |

**Nothing precedes the declaration**, which is the one rule the lead slot has.
Medium, count and kind own it: `Three figures standing in a hotel corridor`,
`1990s vintage anime style cel animation`.

**The default puts framing and angle in the camera module, at the tail.** That
is the opinionated choice, and the evidence is lopsided enough to justify it:
framing leads in two of the eleven, and both are single-subject portraits where
"close-up portrait" is effectively one noun. All three ensemble prompts tail it —
`Shot straight on at chest height`, `Shot on medium format, slightly wide`.

**What decides whether framing can lead is the declaration's head noun, not the
number of people.** The lead rendering fuses the framing word into that head —
`An extreme close-up **portrait**` — so the head has to be a noun that takes an
adjective. It has three forms and only one of them refuses:

| Head | Example | Framing may lead |
|---|---|---|
| count | `Three figures`, `Two young people` | no — a count takes no framing adjective |
| collective | `a couple`, `a crowd of teenagers`, `a family` | **yes** — grammatically singular |
| coordinate | `a young boy and girl` | **yes** |

`A close-up low-angle portrait of a couple` works for the same reason `A close-up
portrait of a young East Asian woman` does, and `A medium close-up three figures
in a corridor` fails. The anime crowd proves it at scale: `densely packed crowd
of teenagers` is an adjective phrase on a collective, leading, with forty people
in it.

So this is not a constraint on ensemble work. It is a consequence of how the user
named the group, and `a couple` versus `two people` is exactly that choice —
which makes it theirs rather than the template's.

### Subjects are members of the declaration, not siblings of it

Every multi-subject prompt introduces the group first and then refers back into
it with a **definite article**:

    Two young people sitting side by side…   ->  The person on the left is…
    …a young boy and girl walking…           ->  The boy, on the left, wears…
    densely packed crowd of teenagers…       ->  central boy… / surrounding students…

That back-reference is the tell, and it means a subject module is **nested**
inside the declaration rather than following it. The consequence for reordering:
moving the group moves every member with it, and members re-rank against each
other *inside* the group — a promotion within a collective changes who the
picture is about without touching where the group sits.

It also means a collective head does not remove the sub-subjects; it changes what
they attach to. `a couple` still needs `the one on the left` and `the one on the
right` to carry two wardrobes, because one collective noun cannot hold two sets
of garments — which is the same one-modifier-depth budget as everywhere else.

Worth promoting when the framing is what the picture is *about* rather than how
it was taken — an extreme macro where the framing is why the subject is
unreadable as itself, a Dutch angle used expressively, a POV. Even where it is
available it stays the exception: two of the six single-subject prompts use it,
and the jester, the sailor girl, the cliff and the collage do not.

**`SHOT_VOCAB` already encodes both renderings, and encodes them inconsistently.**
Framing phrases are noun phrases — `"a medium close-up"`, `"a close-up"` — the
lead form. Angle phrases are verb clauses — `"shot from a low angle"` — the tail
form. The two groups were written against opposite assumptions, and because both
sat in the same `-10` bucket the contradiction never surfaced.

That is the coupled change the module owes: it has to carry **both** phrasings
and pick by position, so the framing pills need their verb form written before
the tail default is real. Otherwise a demoted framing clause is a bare noun
fragment — `a portrait of k3nan. A medium close-up.` Slot and phrasing are one
decision.

*Illustrated* shifts the default: the medium leads and angle tends to follow it
immediately rather than waiting for the tail — `stylized digital painting of a
dark convertible on a winding coastal cliff road, high-angle perspective`.
Framing still tails; `tightly framed medium shot` is second-to-last in the anime
crowd.

Until subjects are structured there is no declaration for the compiler to read,
so today framing and angle sit immediately ahead of the typed text and merge with
it — which produces the one-clause form exactly (`A medium close-up, a portrait
of k3nan`) and is the right approximation while the typed field *is* the
declaration. It stops being an approximation when the declaration has its own
slot.

### What the merge fixes

`_compile_image_prompt()` (`app.py:5770`) carries a lead/tail split: the first
phrase of the `-10` bucket leads, and the rest move to a positive slot behind the
subject. Its own comment explains why — a stack of shot description ahead of "a
portrait of k3nan" spends the model's attention on the shot and renders a scene
the subject is barely in.

The intent is right and the fallback is not. All four image groups share the
bucket `(-10, "list", "visual")`, so **the lead is chosen by vocabulary order
within that bucket** — framing, angle, light, tone. Light is third. With framing
and angle unset, which is the ordinary state, light leads:

    light.golden alone  ->  Lit by low golden-hour sun, a portrait of k3nan.
    tone.noir   alone  ->  In high-contrast film noir, a portrait of k3nan.
    framing + light    ->  A medium close-up, a portrait of k3nan. Lit by low
                           golden-hour sun.

Run against the real vocabulary through `tools/_from_app.py`. The docstring's own
worked example is the third line, where framing is set; the first two were never
considered.

**The leading clause is what the subject sits in.** A framing clause contains it
— `A medium close-up, a portrait of k3nan`. A light clause does not: the light
becomes the thing described and the subject arrives as an apposition to it, which
is mood promoted over its subject. Fifteen of the twenty-nine image-side pills
do this when picked alone, and "golden hour and nothing else" is about the most
ordinary way anyone touches a palette.

The evidence is unambiguous: **light does not lead in any of the eleven
examples.** In the one case it appears early — the cliff painting — it is bound
to a surface and still trails both the subject and the angle.

Splitting the slots is what fixes it. Framing and angle to `-20` because they are
the declaration's own head (`An extreme close-up portrait featuring…`), light and
tone to `30` because mechanisms 5 and 6 put them last. The lead/tail split then
has nothing to do and goes. The difference is not cosmetic: today light's
position is **conditional** on what else you happened to pick, and afterwards it
is **structural**.

### What the merge cannot fix, and why that is the argument for the template

The slot split governs **pills**. The typed field is an unordered blob that can
put anything anywhere, and the compiler is forbidden from touching it —
`_shot_body` closes the sentence and otherwise leaves every character alone,
because rewriting it would change trigger-word case and the user's intent.

So a prompt typed light-first defeats the whole ordering, three ways at once:

    typed alone     ->  lit by direct sunlight, 2 people on a park bench
    + light pill    ->  lit by direct sunlight, 2 people on a park bench.
                        Lit by hard direct sunlight with sharp shadows.
    + framing pill  ->  A medium close-up, lit by direct sunlight, 2 people on
                        a park bench.

Light leads and passes through untouched. The matching pill states the same
property a second time, at the other end. And the framing pill merges into the
*light* clause rather than the subject, so the declaration becomes framing plus
light and the subject arrives third.

None of these is fixable in the compiler. They are fixable only by removing the
place where an unordered sentence can be typed — which is what a slot per field
does. **The ordering rules in this document are advisory while a textarea
exists, and enforceable the moment it does not**, because a lead slot that only
accepts a declaration cannot be handed a light clause. That is the argument for
the template, and it is a stronger one than convenience.

### Two things the merge surfaces

**A confirmation.** The `hardsun` pill reads *"lit by hard direct sunlight with
sharp shadows"* — a light phrase that already ends in shadow behaviour, which is
mechanism 5 arrived at independently. The palette's phrasings were written
against the same encoder and mostly agree with the evidence here.

**A tension worth resolving rather than averaging.** The `practical` pill reads
*"lit by practical lamps visible in the frame"*, which is a statement about the
light. The bathroom's fixtures rendered because they were named as *objects*,
with material and colour, in the place clause — `two white globe pendant lamps`.
Those are not the same statement and neither replaces the other: the pill says
the light comes from visible sources, the surfaces say which sources. A scene
that wants specific fixtures needs both.

### The boundary

Closed vocabulary gets a pill and a glyph; open vocabulary gets a field. The new
slots are open — nobody can enumerate every garment — which is why they are
fields rather than a larger palette, and why the palette does not grow to meet
them.

## The skeleton

The template is not one shape, and it is not a menu of five. It is **emitted** —
the forms below are what one set of rules produces at different sizes, which is
also why an unseen case is a gap in the rules rather than a missing template.

**The order is an opinionated default, and it is the user's to change.** Those
are two statements and neither implies the other. The template **proposes** — it
picks the order below and emits it, with no dialog and no "how would you like
this arranged?", because a question asked before anything exists is a question
asked to avoid having an opinion. And re-ranking is **available from the moment
there are modules**: before the first render, between renders, after. Gating it
on having rendered would be a step that protects nothing.

There is no "nothing to look at" problem to justify such a gate, either. The
compiled prompt is the thing to look at, and it already exists — `/api/compile`
runs the same compiler on a CPU container and the disclosure shows the exact
document that would run. Move a module, watch the document change, in
milliseconds. That is a tighter loop than rendering, so the pre-render case is
the *better* one to support rather than the one to defer.

### The unit that moves is a module, not a clause

A **module** is one thing the picture is about, with everything the prompt says
about it: the declaration, the place, *each subject*, the behaviour, *each
secondary element*, the light, the camera. Reordering moves a whole module.
Promote Subject 2 and its position, anchor, verb, pose, wardrobe and held object
all travel with it — you are moving the character, not a sentence about the
character.

**Inside a module nothing reorders**, because the internal order is not a
preference. The anchor leads because attention is causal; `wear[]` runs head to
feet; the light clause ends in shadow behaviour. Those are binding mechanics. So
there are two levels and they are governed differently:

| | Order | Decided by |
|---|---|---|
| Between modules | user-reorderable | what this picture is about |
| Within a module | fixed | how the encoder binds |

### Three axes, not one — and the earlier version of this section was wrong

This document previously said *"slot order is importance order."* It is not.
Order, extent and the opening slot are three separate effects, and collapsing
them produced a design where the wrong gesture raised prominence.

| Axis | What it controls | The gesture |
|---|---|---|
| **Extent** — words and clauses spent on a thing | **Prominence.** How much the picture is about it. | say more, or less |
| **Order** — which subject is described first | **Left-to-right placement in the frame.** | rearrange |
| **The opening clause** | **The scene's focus** — its own slot, not a subject | choose what leads |

**Order places; extent weighs.** In all four multi-subject examples, description
order matches rendered left-to-right placement exactly — corridor (woman, then
girls), bathroom (red-haired, man, blonde), bench (man, woman), forest (boy,
girl). Word counts track prominence instead: near-equal counts give near-equal
prominence in the corridor (45/40) and the bench (28/27), while the anime crowd's
hero outweighs its collapsed set (24/18).

*Caveat on the evidence:* every one of those prompts also states position
explicitly ("on the left", "in the centre"), so these examples cannot separate
order from stated position. Independent testing is what establishes that order
alone does it.

**What survives is the opening slot.** Moving light ahead of the subject really
does make the light the subject — not because of a general position gradient, but
because the first clause is a distinct slot that sets the scene's focus. The
defect fixed in `_compile_image_prompt()` was never that light led; it was that
**nobody chose it** — vocabulary order did.

**And the consequence that reaches the interface:** the gesture that raises
prominence is *opening a thing and saying more about it*, not moving it. Dragging
rearranges the frame. Those must not be the same control.

### A module is a chip

The representation is already built twice over. A reference chip is a compiled
clause rendered as a manipulable object carrying a role from its own menu; a shot
pill is the same thing with a closed vocabulary behind it; and both live in a
rail under the prompt whose order **is** the order in the document. That is a
module.

So this is not a new surface. `#shot-rail` carries more, the way `SHOT_VOCAB`
carries more rather than gaining a sibling table, and the rule that governs the
rail already governs this one: it costs nothing until it holds something
(`#shot-rail:empty{display:none}`), and a row is affordable when it carries
content and never when it carries one control.

Three consequences follow directly:

- **Dragging rearranges the frame.** Rail order is prompt order is *left-to-right
  placement*, so dragging a subject past another swaps where they stand. It does
  not change how prominent they are — that is extent, and it is changed by
  opening a module and saying more. Two gestures, two effects, and conflating
  them was the error the three-axis table above corrects.
- **The roles are done.** `SHOT_REF_ROLES` is the vocabulary for reference chips,
  for LoRA slots and for modules — one list, three ways to fill it, which is the
  same argument that kept LoRAs off a parallel role table.
- **A subject chip has to open.** A subject module holds an anchor, a wardrobe
  list, a pose and a held object, which is more than a chip shows. The region
  card is the precedent and the shape: touch it and it opens in place, rooted to
  the thing it belongs to, rather than routing to a panel somewhere else.

The last one is where the nesting lands too — sub-subjects are chips inside the
card of the collective that contains them, so `a couple` opens onto the two
people in it.

### The rail shows prominence, and the heat travels with the element

The order is the lesson. A good default, seen every time, teaches the arrangement
better than any explanation of it — light lands at the tail on every prompt the
user writes, and after a while they know light goes at the tail. Nothing has to
say so.

**The field is not static, and that falls out of the three axes.** Prominence
belongs to an element's own content, so rearranging carries its heat with it —
colour moves across the strip and is conserved — while editing changes one
element's temperature where it stands. Those are two visibly different
operations, and between them they teach the model without a word: drag and watch
the heat follow, which says position is not weight; add detail and watch one
element warm in place, which says extent is.

On top of that, **show each chip's weight.** DAAM did this after the fact, by
harvesting cross-attention during sampling and heatmapping each token over the
finished image. The same thing runs *before* a render if the question is
per-module rather than per-pixel: encode the prompt, encode it again with one
module removed, and the distance between the two conditionings is that module's
weight. N+1 forward passes through Qwen3-VL-4B — no DiT, no VAE, no sampling —
and one number per chip, which is the granularity the rail already displays.

So the gradient across the rail is measured rather than asserted, it recomputes
as modules move, and it is right for the prompt in front of you instead of right
on average across eleven examples.

**The display is continuous because the encoder is, and that is not a stylistic
choice.** The obvious-looking version of this is a number or a coloured badge on
each chip, and it is wrong for the same reason a tag list is wrong: it presents
seven independent weights when what exists is one composition whose parts
condition each other. `app.py:936` already makes this argument about the prompt —
*tag lists cannot express binding at all; a sentence resolves it by construction*
— and a per-chip badge would reintroduce exactly that model one layer up, in the
picture of the prompt rather than the prompt.

So the bar is not seven weights drawn as a gradient. It is **the attention field
across the prompt, sampled at chip centres**, which makes the interpolation
between stops meaningful rather than cosmetic. A module is an interval in that
field, not a point in it.

### What the ranking is actually for

Not teaching how the encoder reads. **Surfacing the disagreement between what the
system inferred and what the user pictured, while it is still free to fix.**

The case that shows it: someone describes a character's weathered hands. Every
part of the pipeline behaves correctly — hands are a detail on a subject, they
belong low in that character's clause, and the default is right on average. And
it is completely wrong if the shot in their head is a macro of the hands. Nothing
in the words distinguishes the two readings.

**And it does not cost one take to discover — it costs twenty.** The render says
the result is wrong; it never says why. So what follows is a search: reword,
reorder, add "close-up of", render, repeat. That search isolates nothing, because
editing prose to move a clause also changes tokenization, length and punctuation,
four variables moving together, plus the seed unless it was pinned. Twenty
uncontrolled experiments at two to three minutes each, and at the end the user
has *a string that worked* rather than a rule they can reuse.

Dragging a chip changes exactly one thing. Position moves and everything else is
held — which is the first time ordering is independently manipulable at all,
since in prose a clause cannot be moved without rewriting the sentence around it.
So the user sees hands sitting low and cool before anything renders, drags them
to the front, and the change is a controlled one.

CLAUDE.md already diagnoses this without having a cure for it — *"nudging commas,
swapping a period for a comma, or shuffling clauses around a paragraph to see
what the encoder does differently … a person performing machine work because the
machine will not do it for them"* — and ⌥← / ⌥→ was already the right gesture.
What it lacked was any indication of what a move would do before a take was spent
finding out.

The consequence reaches past time. **Trial and error is the mechanism that
manufactures copy-pasteable prompts.** A string found by twenty renders has to be
kept, because it cannot be re-derived; kept strings get shared; shared strings
converge into a house style nobody chose. Seeing the cause is what makes the
artifact disposable, which is the authorship argument at the end of CLAUDE.md
arriving as a feature rather than a position.

This is Phase 6's **derived or invented, always visible** with a mechanism under
it, including the half that is easy to skip: *every question asked is a small
failure, so pick something, mark it invented, move on.* The system never asks
"did you mean the hands to be the subject?" It ranks them, shows the rank, and
lets the user move it. A visible decision replaces a clarifying question.

**Recursion is what makes the correction cheap.** Because hierarchy nests,
promoting something deep promotes it globally — hands to the top of the
character, character to the top of the shot, and the prompt leads with hands.
There is no "make this the subject" control to design, because *up* already means
that at every granularity.

**Populated by default, at every level.** Opening a granularity shows its modules
already decomposed and already ordered, never an empty form. The user is always
editing a proposal. That is also the real indictment of the empty prompt box: not
that it is blank, but that it never says what it did with what was typed.

**The dependency, which is the whole risk.** The decomposition has to be good
enough to argue with. If "a woman with weathered hands in a green coat" breaks
into modules badly, the user is correcting a parse instead of expressing intent —
worse than a textarea, which at least never misread them in public. That is the
model-reading-intent front half rather than a table matching pills; a table
cannot do this. And it is why the derived/invented marking has to be honest: a
wrong parse must read as *a parse*, so it is the machine's error to fix rather
than the user's words being wrong.

### A bar per granularity, normalized to its own level

Each level carries its own field, recomputed for the modules at that level.

That is what resolves the calibration problem rather than a wider ramp:
**normalize within the level**, and something is always hot, because "which of
these is carrying this" is well-posed inside any set and always has a top. A dead
upper third was the artifact of asking one global scale to describe every depth
at once, and a bar that never reaches hot is claiming nothing here matters, which
is never true of a granularity that exists.

A module therefore has two honest numbers that do not conflict. Hands may be 0.15
of the whole prompt and 0.6 of the character's clause — the first is what the
parent rail shows, the second is what the character's own bar shows. Different
questions, both answered.

It also makes drilling **diagnostic** rather than navigational. A cool shot,
opened, either names the module dragging it or shows everything evenly
weighted — which is its own finding: nothing in this shot is emphatic. That is
the flat stretch an editor scans for, and the same instrument reads it at
sequence, scene and shot depth.

**Bars do not stack.** You are only ever at one granularity, so going deeper
replaces the rail rather than nesting another gradient beneath it. Otherwise a
deep drill becomes a column of competing accents and the one-surface rule goes
with it — an editor at frame zoom is not looking at five timelines.

### Shape at rest, value on click

The field and the number are two different jobs, and splitting them is what lets
the display stay continuous without becoming unreadable.

- **The bar answers *where the weight is*.** Shape only, no figures. It has to be
  legible enough to point at a region, never precise enough to measure one — so
  making the gradient the rail's *ground*, with the chips riding inside the field
  rather than beside a readout of it, is sufficient and a badge is not required.
- **The click answers *what this is and what it is worth*.** The chip opens into
  the module's own template: its fields, the clause it compiles to, and its share
  of the conditioning.

That division is why "a wash" is not a defect. It would be one if the bar had to
carry precision; it does not, and asking it to is what produces the per-chip
badge that flattens the composition back into tags.

It also keeps the number interpretable. A bare `38%` on a chip says nothing —
thirty-eight percent of what, next to what. `38% of the conditioning`, beside the
sentence it describes and the fields that produced it, is a number with somewhere
to go: if it is lower than intended, the fields to change are already open.

One gesture carries both payloads. Click is *edit this module*, and the value
arrives with it — so there is no reveal-the-weights control, which would have
been a mode, and no second place to look. The region card is the precedent and
the shape: touch it and it opens in place, rooted to the thing it belongs to.

This is also where a module's LoRA slot lives. Opening a subject gives its
anchor, its wardrobe list and the `identity` picker together, which is the role
table arriving somewhere rather than needing a surface of its own.

Two limits worth stating, because a number gets trusted further than it earns.
It measures how far a module moves the *conditioning*, which is not the same as
how much it changes the picture. And it is per module, not spatial — DAAM's
actual question, *where in the frame did this land*, still needs something to
sample.

### Position decides form

A module in the lead and the same module at the tail are not phrased the same
way, and the module owns both renderings rather than the user picking between
them. Framing is the clear case:

- **Lead** — fuses into the subject noun phrase, no verb. `An extreme close-up
  portrait featuring…`
- **Tail** — its own clause, verb `Shot`. `Shot on medium format, slightly wide`

So `defining` and `recording` are not a setting. They are what promoting or
demoting the framing module *does*, and the phrasing follows the move.

**Clause count selects the form, not head count.** A collapse merges k subjects
into one clause, which lowers the effective count. The corridor is three people
and *two* clauses — one woman, one collapsed pair — which is why it reads in the
chained-relative form rather than as a triad. Getting this backwards produces a
three-position skeleton for a two-clause prompt, and the empty middle position is
where a fourth figure gets invented.

Two further consequences of emitting rather than choosing: a variable left `off`
removes its slot entirely rather than emitting an empty one, and an `auto` that
derives from another variable recomputes when its source changes.

The slot order never varies, whatever the form:

    declaration → place → subjects → behaviour → secondary → light → camera

Where a form has no use for a slot it is absent, never empty. The zero-clause
form has no `subjects` and no `behaviour`; the one-clause form folds `subjects`
into the declaration. Nothing reorders.

### Zero clauses — the subject is an object, or a structure

No subject clause, no behaviour clause, no gaze. The declaration carries the
object; a `structure` phrase carries how the composition is built, when the
composition is the point.

```
{MEDIUM} of {ANCHOR} {PREP} {PLACE}, {ANGLE}, {RENDER},
{LIGHT} hitting {SURFACE} and {SURFACE},
{SECONDARY} {FRAME_POSITION}, {PALETTE}, {SHADOWS}.
```

> stylized digital painting of a dark convertible on a winding coastal cliff
> road, high-angle perspective, blocky painterly brushstrokes, golden hour
> sunlight hitting rocky orange terrain and green vegetation, flock of white
> abstract birds flying in foreground, blinding bright sun reflection on vast
> ocean, vibrant warm color palette, sharp graphic shadows

Two departures from the photographic forms, both specific to this register.
`{ANGLE}` moves **early** rather than to the tail — third slot, right after the
declaration. And light **attaches to the surface it falls on** rather than
standing alone, which is what `hitting rocky orange terrain` is doing.

### One clause — the subject is the frame

Declaration and subject **merge**. No position vocabulary exists. Place demotes
to a background phrase on the light clause. Everything else is `secondary[]`.

```
{FRAMING} {KIND} featuring {ANCHOR} {POSE}, {WEAR}.
{SECONDARY} {FRAME_POSITION}, {SHADOWS}.
{SECONDARY} {FRAME_POSITION} while {SECONDARY} {FRAME_POSITION}.
{LIGHT_QUALITY} {LIGHT_EFFECT}, {BACKGROUND} in {RENDER}.
```

> An extreme close-up portrait featuring pale, freckled skin and a single blue
> eye wrapped in reflective metallic gold ribbons. Thin gold strips crisscross
> diagonally over the cheek and forehead, casting sharp, hard shadows onto the
> face. Strands of copper hair frame the top edge while the left ear softly blurs
> out of focus. Harsh, direct lighting highlights intricate skin pores and bright
> golden reflections, isolating the brightly lit features against a pitch-black
> background in a bold, high-contrast macro editorial style.

### Two clauses — copular, binary positions

Each subject gets a **definitional** sentence. Positions are left and right only.

```
{COUNT} {KIND} {VERBing} {PREP} {PLACE}, with {SURFACES} {DEPTH_CUE}.
The {NOUN} on the left is {ANCHOR} in {WEAR}.
The {NOUN} on the right is {ANCHOR} in {WEAR}, {POSE}.
Both {GAZE}, {EXPRESSION}.
{SECONDARY} {VERB} {FRAME_POSITION} {POSE}, {RELATION} and {CROP}.
{LIGHT_QUALITY} {LIGHT_DIRECTION} {SHADOWS}.
Shot on {FORMAT} {HEIGHT_AXIS}.
```

`The person on the left is a young man in…`, not `On the left stands…`. Both
two-clause examples take the copular form and both bound wardrobe exactly.

### Three clauses — positional verb, triadic positions

```
{COUNT} {KIND} {VERBing} {PREP} {PLACE}, with {SURFACES} {RELATION}.
On the left, {ANCHOR} {VERB} {POSE}; {PRONOUN} wears {WEAR}, {HOLDS}.
In the centre {ANCHOR} {VERB} {POSE}; {PRONOUN} wears {WEAR}.
On the right {ANCHOR} {RELATES_TO}, wearing {WEAR}.
All three {GAZE}, {EXPRESSION}.
{SECONDARY} {FRAME_POSITION}.
{LIGHT_QUALITY} {LIGHT_DIRECTION} {SHADOWS}.
Shot on {FORMAT}, {COMPOSITION}, {DEPTH_OF_FIELD}.
```

An overlapping subject takes `relates_to` — a clause naming the subject it
overlaps, `presses close behind him with both arms wrapped around his waist`.
That relation is what kept the bathroom's right-hand pair from merging, against
the expectation that they would.

Positions may also **chain relatively**: `On the left stands…` then `Beside her,
to her right, stand…`. That is the form to take when a subgroup sits together.

### Four or more — one hero, everything else collapsed

```
{MEDIUM}, {COLLECTIVE} {PREP} {PLACE},
central {ANCHOR} {POSE}, wearing {WEAR},
surrounding {COLLECTIVE} {SHARED_GAZE}, {SHARED_WEAR},
{FRAMING}, {RENDER_ATTRIBUTES}.
```

Exactly one individuated subject; every other figure is a single collapsed set
clause. Never N clauses.

> …densely packed crowd of teenagers in summer uniforms, central boy with short
> black hair raising a clenched right fist… surrounding students looking in
> various directions, girls in white sailor blouses with green striped collars
> and neckerchiefs, light blue skirts and trousers…

One hero, one set, and it held across forty figures.

### The collapse, at any count

Any subgroup sharing an appearance is described **once**:

```
{POSITION}, {COUNT} {NOUN}, dressed identically in {WEAR}, {SHARED_POSE}.
```

The corridor twins match because the prompt never handed the model two
descriptions to reconcile. Subjects are either individuated or collapsed, and
never mixed inside one group.

---

## Why these shapes

Eight mechanisms, each of which the evidence separates cleanly.

1. **The anchor is the first token of its clause.** Qwen3-VL is causal, so
   everything after the anchor attaches to it. `On the left, a red-haired young
   woman leans…` binds position and hair before a garment token exists. Anchors
   are position, hair colour, or age and size, and must be unique across
   subjects. Three overlapping figures did not merge because all three anchors
   were distinct on all three axes.

2. **The group collapse.** Above. Also the only way crowds work at all.

3. **Material + colour + noun, in that order.** Colour binds strongest, material
   second, and the noun is weakest. `teal suede skirt`, `glossy pink square
   tiles`, `red patent boots`, `orange basin` — all exact. `patterned handbag`
   drifted, because "patterned" specifies nothing and "handbag" is generic.

4. **A pattern named as an object beats a pattern named as a print.**
   `embroidered with small birds` produced birds; `appliqued with bright green
   leaves` produced leaves. The tapestry-print skirt came back generic floral,
   because the floral bomber beside it already owned "floral". One hero pattern
   per figure; everything else demoted to material and colour.

5. **Light is physics, never mood — and it never leads.** There is not one mood
   word in the adhering set — **and this rule stops at the light.** See the
   correction below: on a person the emotional word is the highest-value token
   available. What separates them is whether the abstraction decodes to a
   physical fact ("resigned") or to a preference ("moody"). Every light clause is source, direction and **shadow
   behaviour**, and the shadow clause is the half that binds. It is also last or
   near-last in all eleven, without exception: a light clause placed ahead of the
   subject makes the light the thing being described and the subject an
   apposition to it. Mood belongs to a subject; it cannot be one.

6. **Camera is measurable facts** — format, height and axis, composition, depth
   of field. Never a lens brand, never "cinematic" on its own.

7. **The tail decays selectively.** It is safe for *global* properties — light,
   grade, lens, medium, render attributes — and lethal for anything that must
   bind to a specific noun, because in the tail there is no nearby anchor to
   attach to. `, whimsical woodland creatures` appended to the forest prompt
   added nothing that was not already bound mid-prompt.

8. **Two registers, decided by the first eight tokens.** Photographic declares a
   *scene* and tails into a format; illustrated declares a *medium* and tails
   into rendering attributes. `An anime illustration depicts a young boy and girl
   walking through a lush forest.` proves the two axes are independent — a medium
   declaration in sentence syntax, fully adhering. Sentences are safer whenever
   there is more than one clause.

## The re-mention fault — how subjects merge and vanish

The most destructive failure found so far, and it is invited by the natural fix
for a milder one.

**The milder problem: detachment.** Three subjects, each described once, all
render correctly and at comparable prominence — but a subject with no stated tie
to the others lands in his own sector of the frame. Same location, no visual
error, simply unconnected, as though the other two were not there.

**The natural fix is to elaborate how he perceives them, and it breaks the
image.** Writing *"person c notices her, he doesn't like that she's talking to
person b"* makes people disappear or fuse — one subject arriving as a mutated
extension of another.

The distinction is not cross-reference, because the working version contains one:
*"person a is sitting with person b"* is fine.

| Relation | Example | Result |
|---|---|---|
| **Physical** — contact, proximity, orientation, a shared object | `sitting with`, `presses close behind him with both arms wrapped around his waist` | renders, and **ties the subjects together** |
| **Mental** — perception, opinion, knowledge about another subject | `notices her`, `doesn't like that she's talking to` | subjects merge, duplicate or vanish |

**Why, as far as the mechanism goes.** Attention is causal, so everything after a
subject's anchor attaches to that subject — the rule the whole anchor design rests
on. Naming another subject inside this one's clause therefore creates a *second
attention site* for a character that already has one, and the model either renders
a duplicate or fuses the two. A physical relation is consumed by being rendered;
a proposition about minds has no visual form, so nothing discharges it. Pronouns
are the worst case, carrying no appearance of their own to stabilise which
referent they are.

**A second mention is not itself the fault** — the bathroom prompt names the man
again as `him` and `his` inside the blonde's clause and holds together perfectly.
Two things separate that from the failure:

1. **The relation is physical, so rendering discharges it.** A proposition about
   minds has no visual form, so nothing consumes the reference and it persists as
   a competing attention site.
2. **It points at an individual who was never grouped.** In the bathroom every
   subject enters independently and the tie is declared *on arrival* by the
   newly-introduced subject. In the failure, `person a is sitting with person b`
   binds A and B into a sub-entity before C exists — so C arrives as a peer of
   *the pair*, and relating C to A afterwards means reaching inside an
   established group, which requires breaking its binding to do.

**The rule: relations between subjects are physical, and they are declared as a
subject enters rather than amended in later.** By the time you amend, the
grouping is load-bearing.

**Grouping happens whether or not it was intended.** `A is sitting with B` makes
a sub-entity out of a sentence that reads like a description, and nothing
announces it — the cost surfaces several clauses later when a third subject will
not attach. This is the strongest case in the document for showing derived
*structure* and not only derived content: if the binding were visible when it was
created, unbinding it or bringing the third subject in early would both still be
free.

**Presence and relatedness are separate, and only one of them is a value.** The
detached subject has full presence — correct count, comparable prominence, no
visual error — and no relatedness at all. Extent and order are properties *of a
subject*; relatedness is an **edge** between two, and C-to-B can hold while
C-to-A does not. A linear arrangement can show the first two and structurally
cannot show a graph. Recorded as an open gap rather than a solved one.

**The fix keeps the intent.** A feeling *about* another subject becomes that
subject's own visible state — *"person c watches from across the room, jaw set"*.
One mention, no re-invocation, and the interiority survives intact because a named
emotional state decodes to a morphology (see the emotion correction below). It
also solves the detachment that started it, the way the bathroom prompt does:
a physical tie is what puts someone in a scene with the others, and it is the one
relation in that prompt capable of merging two overlapping figures that instead
held them apart.

This constrains `relates_to` in the subject table: **contact, proximity or
orientation only. Never perception, never opinion.**

## The opening slot makes a character out of whatever is in it

Asking for a *little* perspective produced a long wall with an effectively
infinite vanishing point — further from the target than the neutral version being
corrected. That reads as a badly calibrated model and is not one.

**The model did exactly what it was told.** The correction opened with
perspective, and the opening clause decides the scene's focus, so perspective
stopped being a property and became the subject. The render is a picture *of*
perspective, which is what maximum perspective looks like.

This is one mechanic appearing three times, and only the chooser differs:

| Lead clause | Result | Chosen by |
|---|---|---|
| light | the light is the main character | the author, in an abstract render — **intended** |
| perspective | perspective is the main character | nobody — reads as overcorrection |
| light, by vocabulary order | the light is the main character | the compiler — the defect fixed in `image_slot` |

So the earlier reading of this — that compositional *terms* carry an inherently
extreme prior — was an invented mechanism for something the three-axis model
already explains. The rule is simpler and was already written down:

**Anything in the opening slot becomes a character. A property promoted there
arrives at character strength.** The same word attached to something later
behaves as a property and scales normally.

Which locates the fix precisely. Not "always state geometry rather than
qualities" — that remains good practice for the reasons under *light is physics*
— but **do not let a property lead unless it is meant to be the subject.** That
is exactly what `image_slot: 30` enforces for light and tone in
`_compile_image_prompt()`. The protection exists; what it covers is the question.

Stating geometry still helps for a separate reason worth keeping: a fact carries
its own magnitude — *"the corridor runs away to the right and the far door is
half her height"* cannot be stated without a quantity, while *"dramatic
perspective"* has only the one setting. Use facts where an amount matters, and
keep them out of the lead where the amount should stay moderate.

## Fusion is fine in the lead; apposition is not

"Nothing precedes the subject" is about *competition*, not about the opening
clause being reserved. Plenty belongs there — as long as it is grammatically part
of the subject rather than a rival to it.

| Form | Example | Outcome |
|---|---|---|
| **Fusion** — one noun phrase, subject as complement | `a photograph of ntmo standing outside in a suburban neighborhood` · `An extreme close-up portrait featuring…` · `A close-up portrait of a young East Asian woman` | **good.** The head is the medium, the subject is what it is *of*, nothing competes |
| **Apposition** — two noun phrases joined by a comma | `A medium close-up, a portrait of k3nan` | the framing becomes a second candidate for what the picture is about |

Same words, different grammar, opposite results. The opening clause routinely
carries medium, register and framing in the adhering examples — it carries them
*fused*.

**This is what decides where a framing pill belongs.** `_shot_sentence()`
comma-joins, so the pill machinery can only ever apposition: the one arrangement
in which leading is correct is the one arrangement it cannot produce. Hence
framing at slot 40 with the rest of the camera, phrased to read as a tail clause.

### The same limit again: it cannot weave

Comma-separating clauses is the Stable-Diffusion-with-CLIP habit, and it produces
markedly worse output here than the same content written as prose.
`photograph, lit by a window, low angle, person c` fails where the woven version
of exactly those facts succeeds — this encoder parses grammar, so a tag list asks
it to guess at bindings a sentence would have settled.

The pill machinery emits tag lists by construction. Any `pick: many` group
concatenates its fragments:

    Lit by low golden-hour sun, lit by practical lamps visible in the frame,
    lit from directly above, in high-contrast film noir, on grainy 16mm film.

Five fragments on commas, "lit by" three times. No adhering example does this —
every one of them writes a single woven sentence:

    Even shadowless light with no visible source.
    Hard midday sun from the left throwing short crisp shadows.
    Harsh, direct lighting highlights intricate skin pores and bright golden
    reflections, isolating the brightly lit features against a pitch-black
    background.

So there are **two** hard limits on concatenation, not one: it cannot fuse and it
cannot weave. Neither is reachable by re-ordering slots, which is why fixing
*where* clauses land leaves *how they join* exactly as broken.

Free text does both natively, because someone wrote a sentence. That is the
argument for the storyline in its strongest form: the pills are not a rough draft
of it, they are a mechanism structurally incapable of the form that works.

**And it is what the declaration module is for.** A module's text is free-form and
the compiler never alters a character inside one, so a fused opening written as a
declaration stays fused. That makes it the only place in the system where medium,
register, framing and subject can be a single grammatical unit — a form worth
reaching for rather than a curiosity, and unreachable any other way.

Two routes, no collision:

- **Fused** — write it in the declaration: `A close-up portrait of k3nan.`
- **Recording** — tick the framing pill and it lands at the tail: `In a close-up.`

## What reliably fails

In observed order.

1. **Layering under a sheer garment.** The bralette under the lace top vanished.
2. **Object-to-body attachment.** `a handbag at her hip` became a strap over the
   shoulder. The object appears; the attachment point drifts.
3. **Pose modifiers on a secondary subject.** `one leg drawn up onto the bench`
   did not land; both figures sit normally.
4. **Print-style patterns.** Mechanism 4.
5. **Low-frequency functional objects.** `white ceramic hand dryer` became a
   paper towel dispenser.
6. **Asymmetric per-limb or per-eye detail.** `squinting one eye` squinted both.

All six are one failure at different scales: **a modifier attaching to a sub-part
of an already-modified noun.** Binding is reliable one level deep and unreliable
at two. Subject → garment holds. Subject → garment → what is worn under it does
not.

**Hence the budget: one modifier depth per noun.** Every failure above sits in an
`off` slot in the variable tables, which is not a coincidence — the
low-reliability slots are exactly the ones that have to be asked for.

## Known limits

Eleven examples is a narrow slice — three photoreal multi-subject scenes, five
stylized, three single-subject, all essentially static. The mechanisms above
generalise further than the skeletons do, so the honest split is between what the
evidence *extends to* and what it does not cover at all. **An uncovered case is
recorded here rather than guessed at, because a template that silently emits a
wrong-shaped prompt is worse than one that says it has no form for this.**

### Extends, with the evidence named

- **Object as primary subject.** The convertible carries a whole prompt. `wear[]`
  becomes `features[]` and takes the same `{material, colour, noun}` shape;
  `anchor` falls back to colour and kind (`a dark convertible`), since hair and
  age do not apply. Behaviour drops entirely — there is no gaze clause.
- **Light bound to a surface.** `golden hour sunlight hitting rocky orange
  terrain and green vegetation` attaches light to the material it falls on rather
  than stating it globally. This is the illustrated register's normal form and it
  is why that register tolerates the light clause moving off the tail.
- **Practicals are surfaces, not light.** `lit by a fluorescent tube and two
  white globe pendant lamps` sits in the *place* clause in the bathroom prompt,
  not the light clause, and both fixtures rendered. A lamp you can see is an
  object in the room; the light clause is for what the light *does*.

### Not covered — do not infer a form

- **Zero subjects with no object either.** Pure landscape, architecture, texture.
  The collage is the closest and it is not the same thing: it has a subject and a
  structural rule.
- **Four or more individuated subjects.** The template collapses everything past
  the hero, and that is only validated for a crowd. Four people who all matter is
  a case with no evidence behind it, and the collapse would be actively wrong.
- **Rendered text.** No example contains any. Krea 2 can render text; where it
  goes in this skeleton and in what syntax is unknown.
- **Depth-ordered subjects.** All validated position vocabulary is lateral.
  Foreground/midground/background *subjects* — as distinct from `secondary[]`
  elements — have no slot, and `relates_to` is a relation, not a depth.
- **Motion on a primary subject.** Every photoreal example is static: standing,
  sitting, leaning. `pose` is already the weakest reliable slot, and running,
  falling or fighting is past its evidence.
- **Mixed or conflicting light sources.** One quality, one direction is all that
  has been tested.
- **Aspect ratio.** The template emits the same prompt at 16:9 and 9:16, and the
  composition problem is not the same. Face budget in a wide multi-subject frame
  is a real constraint that lives outside this document.

### The structural rule — a slot the collage needs and nothing else uses

`vintage analog collage, central irregularly shaped snowy mountain range …
structured within a 12x16 grid of square tiles, composition fragments the subject
by alternating tiles with solid azure blue background squares, thin white grid
lines` — a sentence describing how the composition is *constructed* rather than
what is in it, and every part of it landed. It has no home in the skeletons above
and is recorded here because it is evidence that the template's slot list is
incomplete rather than merely untested.

### Trigger words, which this platform hits immediately

None of the eleven examples contains one, and every prompt written on this
platform will. Two rules, both already load-bearing in `app.py`:

- **A trigger word is an `anchor`.** `k3nan` is what a subject clause binds to,
  in the same slot `a red-haired young woman` occupies.
- **Its case must survive emission.** `_shot_join()` (`app.py:5634`) exists for
  exactly this: `k3nan` and `K3nan` are different tokens to the encoder, and
  upper-casing one at a sentence start weakens the LoRA it was trained for. The
  template chooses the separator in front of a clause and never a character
  inside it — so a trigger-word anchor forces the preceding clause's full stop to
  soften to a comma, rather than the anchor being capitalised.

## The correction: emotion is physics too, on a person

The lighting rule was over-generalized into the behaviour module, and the three
Gucci prompts are the evidence against it. They say *"All three face the camera
directly, **expressionless**"* and *"Both look toward the camera, relaxed and
**unsmiling**"* — two of three specify the **absence** of emotion. The
recreations were reported as lacking intent, and that is not an oversight in the
template; it is the template instructing the encoder to withhold interiority.
Composition landed because it was specified. Emotion did not because it was
negated.

**Krea 2 reads its prompt through a language model that learned human expression
from captions of photographs, so an emotional word is a compressed physical
specification.** "Resigned" carries dropped shoulders, lowered gaze, slack mouth
and settled weight — a dozen correlated facts in one token. Writing them out
longhand is both longer and *worse*, because a list can contradict itself — a
clenched jaw with soft eyes yields neither — while naming the state returns a
coherent configuration by construction. Against a fixed attention budget that is
the best trade available anywhere in the prompt.

So the rule is not "no abstractions":

| Abstraction | Decodes to | Verdict |
|---|---|---|
| `moody lighting`, `cinematic`, `epic` | a preference | never — names nothing physical |
| `resigned`, `bracing for it`, `caught out` | a morphology | **always** — a dozen facts at once |

The chain is **want → emotional state → physical configuration → pixels**, and it
is enterable at any point: *"she has just realized he is lying"* is a want, and
the encoder walks the rest. That makes a character's intent the shortest input
with the widest physical consequence in the template, which is why it belongs on
the entity rather than in any one shot's `behaviour`.

`gaze` and `expression` therefore stop being `auto` with `expressionless` as the
default. **A default that negates interiority is not a neutral default** — it is
an instruction, and it is the one the recreations followed.

## The fill rules

1. `count` is a literal number word, and it is the first token.
2. `relates_to` is contact, proximity or orientation. Never perception, never
   opinion: a feeling *about* another subject becomes this subject's own visible
   state. A physical relation may name another subject — rendering discharges the
   reference — while a mental one leaves it competing, and the subjects fuse.
3. **A relation is declared as a subject enters, never amended in later**, and it
   points at an individual rather than into an already-bound pair. See the
   re-mention fault above.
4. Every `anchor` is unique across subjects on at least one of hair, position or age.
5. No colour appears on two subjects.
6. Exactly one `wear` entry per subject carries a `pattern`, expressed as an object.
7. Every other `wear` entry is `{material} {colour} {noun}` and nothing more.
8. `wear[]` emits head to feet regardless of fill order.
9. One modifier depth per noun.
10. `gaze` and `expression` are stated once, for every subject at once.
11. The light clause always ends in shadow behaviour.
12. Nothing that must bind to a specific noun appears after the light clause.

---

## Worked emission

The corridor, filled. Three heads, two clauses. Only `user` values are given;
every `auto` is left to the template.

```
count      3 · kind "figures" · place "a hotel corridor"
surfaces   [{pale blue, floral, wallpaper},
            {honey-coloured, wooden, door frames},
            {deep blue, —, carpet}]
subject 1  anchor "a tall woman"
           wear [purple beret, sheer black lace long-sleeved top,
                 wide studded black belt, teal suede skirt,
                 tall tan leather boots]
           holds {a patterned handbag, at her hip}
subject 2  anchor "two small girls of about eight"
           collapse {indices: [2,3], phrase: "dressed identically in"}
           wear [pale blue short-sleeved dresses with white collars
                 and ribbon belts, white knee socks, black Mary Jane shoes]
           pose "holding hands and standing shoulder to shoulder"
light      quality "Even shadowless light"
camera     height_axis "straight on at chest height"
```

The template supplies `verb: standing`, `depth_cue: running away from the
camera`, `position: On the left` and `Beside her, to her right`, `gaze: face the
camera directly`, `expression: expressionless`, `direction: with no visible
source`, `composition: rigidly symmetrical`, `depth_of_field: sharp from front to
back`, and emits:

> Three figures standing in a hotel corridor with pale blue floral wallpaper,
> honey-coloured wooden door frames and a deep blue carpet running away from the
> camera. On the left stands a tall woman in a purple beret, a sheer black lace
> long-sleeved top, a wide studded black belt and a teal suede skirt, with tall
> tan leather boots; one hand holds a patterned handbag at her hip. Beside her,
> to her right, stand two small girls of about eight, dressed identically in pale
> blue short-sleeved dresses with white collars and ribbon belts, white knee
> socks and black Mary Jane shoes, holding hands and standing shoulder to
> shoulder. All three face the camera directly, expressionless. Even shadowless
> light with no visible source. Shot straight on at chest height, rigidly
> symmetrical, sharp from front to back.

Which is the known-good prompt.
