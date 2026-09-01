# Roadmap, and what it vetoes

Where this is going. Not loaded into a session; read it when deciding whether a
new surface belongs.

The veto list is the operative half and is duplicated into `web/CLAUDE.md`,
because a veto that is not in context when someone adds a panel is not a veto.

---

## Phases

1. Krea 2 LoRA training (musubi-tuner) — done
2. Image inference + datasets and captioning — done
3. Video inference via ComfyUI — done
   - MiniMax-H3: t2v, i2v, first/last frame, ref2va, native soundtrack
   - Wan 2.2 lived here too and was removed — see below
4. Image inference onto the same ComfyUI, with regional multi-character
   LoRA — done. One backend, one image, two GPU classes.
   - A box per character, each LoRA masked to its own rectangle
   - Scene and outfit transfer, when the identity-edit LoRA is downloaded
   - Editing an existing image: "Edit this image" makes a render the scene
     the next one composes into, boxes optional (a LoRA chip arms the
     compose through a conjured full-canvas region), person plates, a
     fidelity number, and a per-box likeness anchor
5. Video LoRA training — **not started, and the trainer is not musubi.** H3
   trains under AI Toolkit; see below.
6. **The Dynamic Canvas** — next, and sketched rather than specified below

The end state is one application where a generated still flows into a clip
without a round trip through the filesystem — the "Animate" and "As reference"
buttons on a finished image are the first piece of that.

### Phase 5 — video LoRA training, and the trainer it needs

**Wan 2.2 was the target and Wan 2.2 is gone.** It was tabled here on a bill —
musubi cannot train the `fp8_scaled` weights this platform downloaded, so it
wanted a second 53 GB copy of the 14B pair at bf16 plus an 11 GB bf16 T5, about
64 GB of duplicate weights on a volume whose only way to reclaim space is the
Modal CLI. That bill never got paid, and then the whole family was removed for
being a second of everything — see "One video family" below.

So the phase survives and its shape changed completely. The video side is H3,
and **H3 trains under AI Toolkit rather than musubi.** That is the fact that
matters, because the Wan plan was a *rename* — `train_job` is already the exact
three-step cache-latents / cache-text / train shape, and only the script names
would have changed. A second trainer is not a rename. It is a second image, a
second set of pins, and a second recipe to keep working.

Two things follow, and both outlive the decision to wait:

- **"Downloaded" and "trainable" are different questions**, and they were
  different for Wan too. A Train surface offering video cannot check `present` —
  it has to check for whatever the trainer actually loads, or it offers a run
  that dies after the dataset is cached.
- **The dataset side is already done.** A set counts its clips, the sidecar
  layout is identical (`{clip}.txt` beside `{clip}.mp4`), and nothing about the
  storage contract changes for whichever trainer arrives. That was true when the
  target was Wan and it is still true now, which is the argument for having built
  it that way.

### Phase 6 — The Dynamic Canvas

The canvas becomes where the work is done rather than where the result appears.
Region prompts and LoRAs set on the canvas. Every attachment a drop — a photo
onto a box is that character, a photo onto the frame is the scene, and the video
side's keyframes and reference pictures arrive the same way instead of through a
tray of 32px tiles. One gesture, one place, for every picture a model can be
given.

What you touch answers with its own affordances, at the place you touched it,
and only the ones that object can respond to. Not a menu at a fixed corner: that
is a panel in miniature parked at a coordinate, and it puts the same affordance
count on every object regardless of what the object is. The numeric escape hatch
— X, Y, W, H — stays reachable behind one of those affordances rather than
beside them, because it is precision work you occasionally need and a drag
cannot give.

**The region bar and the map are gone as of the canvas-native rework above**,
and the question below is answered: the map went with the row rather than being
rehomed, because the boxes still come off a finished render and two cheaper
things already lead back to them. The rest of this section stands. It
eliminates the region bar, which raised the question worth settling first,
because the answer decides whether this is a move or a deletion: of the three
things in that bar only one is regional. The scene and outfit plates only exist
when regions are armed, so they follow. The region prompt becomes canvas-native,
which is the point. But the map exists *because* the boxes hide on a finished
render — if the dynamic canvas keeps them legible over a result, the map has no
job left and should go with the row rather than be rehomed out of habit.

**Why the app does not already work this way.** The platform was built in a
familiar idiom on purpose — a console, a Generate button, panels, modes —
because the guts had to exist and be understood before a new surface could be
imagined on top of them. Driving ComfyUI, the job/status/stop contract, the
volume layout, regional masking, the prompt compiler, LoRA training: none of
that is guessable from a mockup, and designing a novel interaction over a
backend you do not yet understand produces a surface that cannot be built. So
the conventional version came first, and it is scaffolding rather than debt.
Read the console that way — it is what made the rest legible, and it is expected
to go.

**Part of this waits on models, not on effort.** Three tiers, with very
different costs:

- *Reachable now.* Touch-to-select and touch-to-edit. Segmentation on a finished
  render turns flat pixels into addressable regions, and regional inpainting
  re-renders one mask. The regional path here is already element-as-component in
  primitive form; the gap is that boxes are author-declared rather than
  model-derived, which is a segmentation step and not a research problem.
- *Reachable but expensive.* Elements that persist across renders. A LoRA gives
  that for a character, which is why this platform trains them — but making it
  general means a LoRA per element and a training run per thing kept, which
  changes what a library is: weights, not images.
- *Not there.* A frame computed from a spatial arrangement rather than authored,
  and a camera with real physical limits. That is causality between a 3D scene
  and a 2D image, which diffusion does not model. It needs a world model, or a
  hybrid where real 3D drives conditioning, and nothing off the shelf does it.

The consequence: the interaction is mostly reachable and the physics is not, so
this phase should chase the first tier and leave the third alone until the model
exists. Attempting it, failing, and concluding the whole direction is fantasy is
the specific mistake this paragraph exists to prevent.

### The Playground, and where the veto line actually runs

There is a node room behind a header door now, and it needs a paragraph here
precisely because the closing line of this file names "a node graph with
better typography" as the failure to guard against. The line the vetoes draw
runs around the **product canvas** — the surface where somebody makes
pictures — and the Playground sits outside it on purpose: a lab, entered
deliberately, where the graph our builders write is the thing on screen. It
is not a second backend (same engine, same containers, same job contract, a
new caller), and nothing in it ever promotes into the console — no workflow
picker grows on the composer, no node leaks onto the canvas. The two sanctioned
crossings are quiet: outputs land in the shared gallery with their graph
embedded in the file, and the model menu can run a saved workflow in the
built-in graph's place — with the console still compiling the prompt, so the
"prompt is a compilation target" veto holds even while the graph is yours.
The month-four failure is still the month-four failure: the day a Playground
concept looks like it wants a *panel* on the product canvas, it goes through
this file's test like anything else.

### Where this is going, and what that vetoes

Phase 6 is the first move toward an end state worth stating in full, because
every one of these exists to kill a specific cheap fix — and the cheap fix is
always a panel. If one of these has never vetoed anything, it is not earning its
place.

**If a gesture cannot communicate intent without a label, the answer is a better
gesture or a smarter reading of it — never a control panel.** Every other rule
here is an instance of this one. It also kills the tutorial: onboarding, coach
marks, first-run overlays, the little animated hand. Those are labels in a
costume. A gesture that needs teaching is the wrong gesture.

**The canvas never changes.** Character, wardrobe, action, location, camera —
one surface. What changes is what the surface is willing to hear. Kills tabs,
rooms, modes, a spine, a back button. Today's Generate/Train/Datasets split is
on the wrong side of this.

**What you touch decides the mode.** You do not enter character mode, you touch
a face. Mode is a consequence of attention, never a precondition for it. Kills
tool palettes, mode switchers, a selected-tool state, anything you must do
*before* the thing you meant to do. Ambiguity resolves toward the smaller
object — a hand over a counter is the hand; widening is cheap and obvious, while
guessing large silently edits the wrong scope.

**Everything is live, nothing is labelled.** Every element addressable at all
times, none of them drawn as a control. This is the hardest engineering in the
product and it exists so the screen can stay empty. Kills inspectors, sidebars,
property panels, toolbars, persistent chrome — which is what the console is.

**Nothing asks for confirmation.** Move something and the frame changes. Kills
modals, "are you sure", staged edits, and the Generate button. The trade is that
undo must be total, because reversibility is what replaces confirmation.

**Duration starts at zero.** A still is the default and time is added. Someone
who wants one image should finish and leave without learning that motion exists.
Kills a timeline on first run and any flow treating a photograph as the
degenerate case.

**The frame is computed, never authored.** Arrangement produces the frame; you
never compose by dragging contents around a viewport. This is what makes direct
manipulation honest — you move the world, not the picture of it.

**The camera is a character.** It has a position, a path, a want and real limits.
It can be late, it can look away, it can be wrong. Kills the camera as a settings
group and lens choice as taste.

**Derived or invented, always visible.** What was read from your words is marked
one way; what was filled in for you is marked another and is cheap to reroll.
This is the entire trust surface and it needs no dialogue. Kills the chat panel,
the assistant sidebar, and clarifying questions — every question asked is a small
failure, so pick something, mark it invented, move on.

**The library is closed until you reach for it.** A drawer, not a workspace.
Browsed, not searched. Things are applied, never imported: edits in a scene are
scene-local, edits in the library propagate, and those are two different acts in
two different places. Kills a persistent rail, a docked browser, an asset
manager.

**It never remembers unless told.** Same words, same result, a year later. Kills
personalisation, suggestion engines, learned preferences, recently-used
defaults, anything that pre-populates. The first screen of the thousandth
session is the first screen of the first.

**On touch, and the trap in it.** Designing for a tablet is a constraint that
does the work for you: no hover kills progressive disclosure, no right-click
kills the hidden second layer, imprecise input forces you to manipulate objects
instead of widgets. But that argument runs ahead of the tools — keyboard and
pointer are first-class here and will be for a long time. What it is really
guarding against is a feature that works *only* with a cursor, because that is a
fork, and the touch half of a fork becomes the degraded half within two
releases. Design to the coarser input; let precision be a bonus. Hover earns its
place for exactly one thing — showing what you would be touching before you
touch it — which is spatial feedback about scope rather than a control being
revealed. "No hover" is not "remove the shortcuts".

**A prompt is a compilation target, not something the user writes.** Nobody
should have to learn a text encoder. Every model arrives with its own grammar —
H3 wants a six-field document, Krea 2 wants prose with the camera clauses
dropped, and the silent family that used to sit between them wanted plain prose — and asking a person to hold three formats in their head
so a checkpoint can be fed correctly is the tool making its problem into theirs.
The user says what they want; the app knows what each model needs; the prompt
itself is an implementation detail they should never have to see.

Half of this is already built and is the proof it works: `SHOT_VOCAB` is one
vocabulary with three compilers behind it, and nobody types
`integrated_multimodal_description:`. Replacing the prompt field entirely — a
model reading intent instead of a table matching pills — is the same idea with
a better front half, and the direction this is going.

Reproducing a prompt is not worth protecting in the first place. A prompt is a
guess at how to speak unnaturally to one text encoder — an artefact of which
checkpoint you happened to be feeding on the day, with no value of its own. The
same intent producing a different prompt twice is not a defect to engineer
around; it is what a conversation is, and the second prompt is no less arbitrary
than the first.

**What is worth keeping is the intent, and that is what the sidecar should be
read as recording.** `prompt_typed` and `shot` are the durable half — what you
actually meant — and the compiled `prompt` is a receipt: this is what this
encoder was told, this once. Reuse, Copy and the metadata sheet prefer the typed
one, which reads as a legibility choice and is really the deeper one.

The gallery itself shows neither, and never has: a card carries the picture, its
kind and its age. Worth saying because "show the prompt on the card" is an
obvious-looking addition and it is wrong twice over — a six-field document is
not readable at thumbnail size, and a prompt is an implementation detail of
whichever encoder was being fed that day.

It also pays off the moment a model is replaced. Intent recompiles for whatever
comes next; a stored prompt is worth nothing to a checkpoint that wants a
different grammar, and every prompt in the gallery would otherwise be tied to
the encoder it was written against.

**And that is the actual shape of it: you argue with the picture, you do not
author a prompt.** "Make her older." "The light is too warm." "No, the other
one." Iteration is conversational and corrective, the way any real working
session is — including the ones that produced this file. What nobody should be
doing is nudging commas, swapping a period for a comma, or shuffling clauses
around a paragraph to see what the encoder does differently. That is a person
performing machine work because the machine will not do it for them.

The tell is in how AI writing is caught: it reads as machine-made because it is
*too smooth*. People think and write in bursts — fragments, a correction, a
second thought that contradicts the first, the real point arriving third. A tool
that demands one clean flowing paragraph is demanding that a human produce the
artefact of a machine before the machine will listen. Backwards.

So fragments are the expected input, not the degraded case. Out of order,
incomplete, self-correcting, arriving in pieces over a minute — that is the
normal shape of intent and the system's job is to take it that way. The prompt
that comes out the other side is the app's business, and stays in the sidecar so
a run can always be replayed.

**Why this matters beyond ergonomics: it puts authorship back in the work.**
The industry built extraordinary visual technology and then made it usable only
by engineers. Prompt engineering is a programming skill wearing an artist's
clothes, and it produced a strange artefact — a prompt is copy-pasteable, so
someone else's method transfers whole. Take their string, get their results,
generate endless variations in their voice without ever having their taste. That
is what hollows out AI-assisted work: the authored thing is a text file, and a
text file is a tool anybody can pick up.

The proof is public. Every image on Civitai looks the same, and it is not
because the models cannot do anything else — it is because the prompt is the
method, methods get shared, and a shared method converges. The same incantations
get pasted in front of every subject, the same negative boilerplate behind it,
the same dozen checkpoints and LoRAs underneath. The result is a house style that
nobody chose and nobody authored, arrived at by thousands of people
independently copying each other's strings. That is what an aesthetic looks like
when its unit of authorship is copy-pasteable.

A conversation does not transfer that way. Not because it cannot be copied —
anything can be copied — but because copying it gives you a record of somebody
else's session rather than an instrument you can point at your own. It is long,
situated, full of corrections that only make sense against what was on screen at
the time, and useless without the reactions that produced it. Nobody has the
same argument twice, including the person who had it.

**The strongest version of this is not conversation, it is that the instrument
is made of your own material.** Authorship at the prompt is thin — it is the
last few centimetres of a pipeline somebody else built. Authorship in something
trained or conditioned on your own photographs is the model itself learning what
you meant, and it is why training is Phase 1 here rather than an advanced
feature bolted on later. The trainer is in the application, the picker reads the
volume rather than a registry, and a dataset is a folder of your images with
your captions beside them. That shape is not an accident and it is not for
convenience: it is what a tool looks like when the person using it makes their
own instruments.

**This paragraph used to say "owning the weights", and that sentence was the
vehicle wearing the principle's clothes.** Every argument above it is about one
thing, and the direction matters: **the unit of authorship must not be
transferable.** A prompt fails that test precisely *because* it copies whole —
paste the string, get the look, bring no taste. A LoRA passes because it is not
a string. **And a set of your own photographs passes for exactly the same
reason** — copying it hands somebody your pictures, not your eye.

Stated the other way round this reads as a virtue, which is how the sentence
went wrong the first time: "owning the weights" sounds like a claim about
possession when it is a claim about *non-transferability*. The weights were one
way to hold your own material, not the reason it mattered, and promoting them to
the headline made a rule about a file format out of a rule about authorship.

That is not a pedantic distinction, because the format is already moving.
Personalisation is shifting from training toward reference conditioning —
IC-LoRA, ID-LoRA, multi-reference identity anchoring, a reusable character
object binding reference sets and wardrobe across models. `H3_CAST_KINDS` and
the cast buckets are already that object: slots holding *your* pictures, bound
to a person, read by whichever model is in front of them. Same authorship,
different vehicle. LoRA becomes the self-hosted tier rather than the definition.

**So the veto is about instruments, not about weights.** Nothing in this
application browses, searches or recommends somebody else's instruments —
weights, reference sets, character bibles, style packs, or whatever the next
format is called. Written the old way it had a hole exactly where the drift
happened: a "browse character packs" surface contains no weights at all, would
pass the letter of the rule, and would undo the paragraph above completely. It
is the same feature that looks obviously helpful in a pull request. Pulling a
specific file you were sent is fine — that is what the Drive route is for. A
marketplace is not, whatever it is a marketplace *of*.

Which is where this stops being a philosophy and becomes the file above: if
intent is the durable record and the prompt is a receipt, then what the sidecar
keeps *is* the authorship. The technical decision and the artistic one are the
same decision, and that is the strongest argument that the technical one is
right.

**The failure to guard against is not the model, the latency or the scope.** It
is month four, when something does not fit cleanly and the cheapest fix is a
panel. That is the moment this becomes a node graph with better typography.
