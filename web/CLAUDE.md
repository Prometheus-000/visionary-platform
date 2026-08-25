# The front end

The UI rules for `web/`. These moved out of the root `CLAUDE.md` so they load
when somebody is working under this directory rather than in every session; the
argument, the vetoes and the measurements are unchanged from where they sat.

Everything universal — Philosophy, Conventions, Storage, Phases and every
standing veto — is still in the root file, and this document assumes it. Where
the prose below says "this file" or names Phase 6, it means the root `CLAUDE.md`:
these sections were written inside it and the references are left as they were.

## The page

The UI is not organised the way the root file is. There are three subsystems
and two domains, and the page follows the domains.

- **One canvas at a time, and a batch is frames of film rather than a contact
  sheet.** A batch of four used to be a two-up grid at half height each, and
  that arrangement is what broke the regions: one set of boxes applies to the
  whole batch, so there is no single surface to draw them on, and they landed on
  the first cell of the four you were trying to compare. Drawing them four times
  would have said the wrong thing; drawing them once said it about one arbitrary
  cell.

  So each result fills the canvas and `‹ 1 / 4 ›` steps between them, one
  sliding out as the next slides in. The result on screen *is* the canvas —
  which is the shape the rest of this file has been heading toward anyway, and
  the surface an inpaint acts on when there is one. Every frame stays mounted
  and the track is a transform, so stepping costs a compositor frame and nothing
  re-fetches.

  It also retires a rule rather than leaving it to rot: the still on the canvas
  used to open the lightbox on click, because *"a result you have to send to the
  gallery to look at properly is a result the canvas is only pretending to show
  you"* — true when a batch was four half-height thumbnails, and answered
  directly by showing every result at full canvas size. What replaced the click
  is one rule with no geography in it: **a click addresses whatever is under
  it** — over a region it opens that region, over bare picture it does nothing.
  Full screen is the expand button and Space. The alternative was a click that
  meant two different things depending on invisible geometry, which hover can
  disclose on a desktop and cannot on glass.

- **A render is replaced when the next one lands, not when it is asked for.**
  Pressing Generate used to blank the canvas immediately: `start()` reset the
  whole run record, so the thing you were judging disappeared for the length of
  the run and came back as something else. On the image side that is seconds; on
  the video side a take is two to three minutes, and the shot you were deciding
  about is gone for all of them — at exactly the moment you wanted to compare.

  So the run record carries two ids. `jobId`/`files` are the render *on screen*,
  moved only by `finish`, and `runId` is the job being polled. They are different
  renders for the length of a run, which is the whole point: `fileUrl(jobId, f)`
  keeps addressing the old job's bytes while the new one is sampled. A run that
  fails or is stopped leaves the picture up too — a request that never produced
  anything should not take away the last one that did.

  What reports the run instead is a hairline along the top edge of the canvas.
  The full centred bar is for a cold start, where there is nothing to keep.

  And when the render is a clip, it *plays* when it lands — muted, looping. The
  `muted` is load-bearing, not a taste: every browser refuses unmuted autoplay
  without a gesture, so removing it (say, to honour H3's soundtrack) does not
  produce autoplay with sound, it produces a 200s render arriving as a strip
  parked at 0:00 — the product's best moment spent asking for a click. The
  soundtrack is one tap away on the native controls.

- **The canvas is the largest thing on screen, always.** Options live in a bar
  under it, never a rail beside it: a settings column costs the picture 384px
  of the one dimension it cannot get back, and vertical is the cheap axis. The
  bar is capped so its fullest state cannot push the canvas out of frame, and
  anything sized to fit the canvas measures the canvas — a `dvh` sum is wrong
  the moment the bar grows.

  **"Vertical is the cheap axis" is false on a laptop, and the measurement is
  here so nobody re-derives it.** Fitting each render aspect into the canvas at
  1512x982 leaves this much unused:

  | render | dead ↔ | dead ↕ |
  |---|---|---|
  | 16:9 | 152px | **0px** |
  | 4:3 | 513px | **0px** |
  | 1:1 | 735px | **0px** |
  | 3:4 | 908px | **0px** |
  | 9:16 | 1068px | **0px** |

  The picture is height-bound at *every* aspect, so the bar always comes out of
  the picture, while four of five leave enough horizontal room for a rail twice
  over. On a tablet in portrait it inverts — 834x1194 leaves 297–469px vertical
  on landscape renders and nothing on portrait ones — so neither placement is
  right everywhere and the axis that is free depends on the render, not on
  taste. What is *not* an option is floating the console over the picture: it
  was built and rejected, because a backdrop blur means judging a render through
  something, and its legibility then depends on what you generated.

- **The console is sorted by how often you reach for something, and by nothing
  else.** The row holds what you touch constantly — the dimensions, the LoRAs,
  the shot pills, the regions. One button holds what you touch rarely — the
  model, the sampler, steps, CFG, shift, the seed, the batch count.

  The axis is frequency, and the near miss worth recording is *scope* —
  per-generation against per-session — because it sounds right and fails on its
  own examples. Almost nothing in the row genuinely changes every take: you do
  not pick a new aspect ratio per render. And CFG is not a thing you set once a
  session either; it is a thing you almost never set. Scope does not predict
  where a hand goes. Frequency does, and it is the one you can answer by
  watching yourself work.

  It also dissolves the case that looked like an exception under the other two.
  The seed *is* different on every render, so by scope it belongs in the row —
  but nobody types a seed. You draw a random one, see something worth keeping,
  and take it off the result. Rarely reached for, so behind the button, and no
  special pleading required.

  The button is named for the model, because the model is the rarely-touched
  choice that decides what every frequently-touched one *means*: which sizes
  exist, whether there is a negative prompt to write, whether LoRAs load at all.
  Naming it `8 steps · CFG 1.0` named it after two values nobody chose — those
  are the checkpoint's defaults. The checkpoint is the choice; the numbers are
  its consequences.

- **The console has a budget, and the prompt is what yields to it.** 30% of the
  viewport. Everything else in there is fixed or conditional — the strip is one
  row, the rail appears with the first pill, and the boxes cost it nothing —
  so the prompt field is the only part that grows without asking, and measuring
  showed it was also the part that broke the budget alone: at a flat 168px cap
  the worst case was 39.8% of a 1440x900 window, 136 of which was the field.
  `fieldMax()` hands it whatever is left, down to a two-line floor, because a
  budget that wins by making the prompt unusable has optimised the wrong thing.
  A ResizeObserver on the console is the half that makes it true more than once:
  arming regions and picking pills both happen long after the last keystroke,
  and without it a long prompt sat at 30.0% and climbed to 38.1% when they
  arrived. It converges in one pass because `fieldMax` subtracts the field's own
  height, so `other` does not move when the field does.

- **The note line under the strip is reserved, and Generate does not move under
  your finger.** The warnings (`#console-notes`) used to mount on demand, which
  read as the obvious economy — a line that says nothing should cost nothing —
  and missed where the cost went: the console is pinned to the bottom of the
  stage, so a note appearing shifted every control in it by one line, at
  exactly the moments a hand is over Generate (the LoRA notes arrive
  mid-typing, the keyframe note mid-attach). One 18px row held at rest is the
  price of a button that stays where you aimed; `fieldMax` absorbs it, so the
  30% budget still holds at every viewport. Two spans share the row, and only
  a second line — both notes at once, or one wrapping — still moves anything.

- **Typing with nothing focused lands in the prompt, not in the hotkeys.** A
  click that misses the field by a few pixels leaves focus on the body, and a
  sentence typed next used to arrive as a hotkey barrage — every space toggling
  the full-screen viewer, every backspace one keystroke from clearing the
  canvas. The first stray letter now focuses the visible prompt and lands in
  it; everything after types natively. Letters only, and the focus is
  synchronous: Space keeps its full-screen meaning when it is the *first* key
  pressed, because a sentence never starts with one — and a focus deferred a
  frame (the `applyWrite` pattern) let the whole sentence arrive as stray keys
  while it was still in flight.

- **An icon invites, a tooltip names, a click teaches — and for a control whose
  home you are already in, that is enough.** The rule survives; what changed is
  that neither of the two things it was being used to justify was a control.

  `#g-shot` and `#g-regional` were 34px glyphs in the settings strip, side by
  side, and together they flattened the two biggest features in the app into one
  confusing pair — with a count riding half-outside the regions button that read
  as an error pip rather than "2 regions". Both were doing the one job an icon
  cannot do: announcing a capability to someone who does not know it exists. The
  strip is a row you scan when you already know what you want, which is why an
  icon works there for a *control* and fails there for a *destination*.

  So each took a word, and one of them moved. Regions is a canvas verb — you place
  a character by drawing on the empty frame, and the empty canvas says so in words,
  once, where the attention already is, so it left the strip entirely. Shot stayed
  where it was and gained the word it was missing: it writes into the prompt, and
  `+ LoRA` beside it is the same kind of door. Giving it a row of its own was tried
  and cost 34px at rest for one button — see the palette note below. What is left
  in the strip are controls and two doors that say where they go, which is what the
  rule was always about.

  It does not generalise past the row. `#ins-toggle` keeps "Captions" because it
  wears the same sliders glyph as two unrelated panels, so its icon is not
  identifying anything, and the hyperparameters keep their names because a bare
  number is not a value. The rule is that an icon can carry a control whose glyph
  is unambiguous *and* whose home you are already in.

- **A utility lives with the controls; navigation lives in the top corners.**
  Reaching the gallery is part of making something, like writing a prompt — so
  the last generation is a thumbnail beside Generate, 15px off the bottom, where
  the hand that just pressed Generate already is. The Camera app puts the last
  frame next to the shutter for that reason and it was read wrong once here:
  the pattern was copied into the header, which put the two halves of one loop
  at opposite corners of a 1194px screen.

  Moving *between* the gallery's layers is navigation, and it belongs in the top
  corners from the moment you are inside — not promoted there after a
  bottom-centre pill hands you over, which is what a "View all generations"
  button did and why it read as inconsistent. There are three layers, and the
  third is easy to miss: the paged view you swipe, the grid, and the picture
  with its chrome off, which is entered by *tapping* rather than swiping. A tap
  used to close the viewer, so leaving was doing double duty for looking
  closely and every attempt to see a render properly dismissed it.

- **The cell is the picture's shape, not a box the picture is fitted into.**
  Thumbnails were always `contain` rather than `cover`, because cropping throws
  away the thing you opened the gallery to see — and `contain` inside a fixed
  4:3 cell pays for that in bars: a portrait render arrived smaller than the
  space spent on it, and a set of portraits was two thirds `--wash-2`. The box
  was the half of that decision that cost. It is gone on desktop, and the cards
  are packed into columns instead, so nothing is letterboxed and nothing is
  cropped.

  **It is twenty lines rather than a dependency, and the reason is reading
  order.** Every cheap way to do this reflows it. CSS `columns` fills a column
  top to bottom, so a listing that is newest first becomes newest *down the
  left edge* — and the top-left card being the last thing you made is the
  gallery's whole contract. The React packages that keep the order distribute
  by `i % cols` and give up the packing, so the columns end up as uneven as the
  pictures are. The ones that pack properly measure the DOM, which means
  reflowing as each image decodes. Native `grid-template-rows:masonry` is still
  being re-argued as `item-flow` and is not something to ship on yet.

  A greedy shortest-column pack has the property none of those has: with every
  column starting empty the first row *is* items 1..n left to right, and each
  item after that lands in the column that is currently shortest, so it reads
  approximately row-major. It is also stable under paging — greedy over a
  prefix is the prefix of greedy over the whole list — so appending a page
  never moves a card already on screen.

  **Nothing is measured, because the server already knows.** `/api/gallery` and
  `/api/dataset` carry the pixel dimensions of every item, so the layout is
  decided before a byte of picture is fetched and a card with a slow cover does
  not move the ones around it. The fallback for a result with no sidecar is the
  4:3 box it always had, and the fallback is written in both places on purpose:
  a packer and a stylesheet disagreeing about one card is a gap under it.

  **It stops at 1024px.** Below that the grid still crops to squares — the one
  place this app trades information for density — and the two layouts are
  different DOM rather than two stylesheets, because a masonry's column
  elements would hand that grid its cards in chronological order *down* each
  column. There is no CSS that unpicks that: `display:contents` on the columns
  flattens them into the grid in column order, which is the wrong order rather
  than no order.

- **The small screen is the design tool, not a port of the big one.** Three
  faults this session were live on desktop for months and only became visible
  under a phone or tablet: `appearance` had never been reset on any select, so
  desktop had been rendering macOS arrows as its chevron; the viewer's drag was
  broken on *trackpad* specifically, because `<img>` is natively draggable and
  the resulting HTML drag fires `pointercancel` — touch was the case that
  worked; and the gallery thumb was in the wrong corner at every size. Below
  1024px the layout stacks and the grid crops to squares, which is the one place
  this codebase trades information for density, and the trade is right because
  there the screen is the constraint rather than the design.

  **And the other half, which this rule reads as licence for if you stop at the
  first sentence: the small screen is the right place to find a fault and the
  wrong place to decide every screen has it.** Two changes made for a phone
  reached desktop by accident and both were wrong there. The last-generation
  thumbnail was declared outside every media query, so it sat beside Generate at
  1512px — where the drawer is already a column next to the picture and the
  header button already opens it, making it a second door onto a room that has
  one. And the viewer lost click-away-to-close, because a tap meaning "show me
  more of this" is right on glass and overwrites a convention on a mouse old
  enough that its absence reads as a broken dialog.

  Both were found by grepping for rules declared outside a media query, which is
  worth doing after any pass that starts on a phone. Where the two genuinely
  differ, split on the *pointer* rather than the width — `(hover:none)` is
  asking the question the layout actually cares about, and a tablet with a
  keyboard is neither of the things a width test thinks it is.
- **The seed rolls until you type one, and nothing writes it back for you.** It
  used to: `finish()` put the run's seed into the field when a document existed
  and the field was blank, so that editing one assumption moved only what that
  edit implied. It was gated on the document precisely because the ungated
  version is a trap — *a seed that stopped rolling for a person who never
  engaged reads as "Generate is broken, it keeps making the same picture"* — and
  with the documents deleted the gate can never be true, so the behaviour went
  with them rather than being ungated by default.

  **What made that safe to lose is that the seed is on the render.** It is drawn
  once, sent to the sampler, written to the sidecar beside the file, shown on the
  metadata sheet and restored by Reuse. A generation knows its own seed forever;
  what is gone is only the input box filling itself in.

  **The condition for bringing it back is a surface where editing an image is
  the thing you are doing**, and their reasoning is what names it: the question
  a pin has to answer is *"is this person iterating, or starting something"*,
  and the document was a bad proxy for it — it meant "you used a feature", not
  "you meant to hold this frame". An edit surface answers it by construction.
  Changing one thing about a picture that already exists cannot be the start of
  something new, so there is nothing left to infer and the seed simply holds.

  There is no such surface yet. Inpainting is sketched under Phase 6 and the
  krea2edit compose exists for scene and outfit plates, but neither is somebody
  sitting in front of a render changing one thing about it. **Until that exists,
  a pin has no honest trigger** — which is why this is deleted rather than left
  behind a flag.

  **And a pin is never cleared when the size or the model changes.** Considered
  and rejected, and recorded because it looks like a bug for as long as it is
  not. Same seed, wider frame is a comparison people actually run; clearing
  deletes it. It would also be the first control here that silently empties
  another field — `setImg`/`setVid` are shallow patches, and the only writes of
  `seed: ''` are the two Reset buttons, which is a person asking. A value that
  empties itself when you touch a control you did not think was a decision is
  the specific way a surface stops being believed. The number is visible in its
  own field and one gesture from gone, and that visibility is what makes
  clearing it for you unnecessary rather than merely risky.


- **Generate is the page, not a destination.** It has no nav item. Train is one
  door, labelled with where it leads rather than where you are, so two things
  never look equally selected. It carries the training run's progress, because
  a run lasts hours and you are meant to leave and keep working.
- **Image and video share the canvas and the gallery. The composer is
  per-kind.** This used to say the prompt survives the switch too, because a
  shot described as a still is the same sentence you would describe as a clip.
  True of the sentence and false of everything around it: a Krea 2 LoRA is not a
  H3 LoRA, and carrying one across loaded it into a run that could not use it —
  silently, because the note warns about names that resolve to nothing and about
  models that read no LoRAs at all, and a LoRA for the wrong architecture is
  neither. Prompt, negative, the pill rail and the chips now swap with the kind,
  and both buffers are kept: switch away and back and everything is as you left
  it, which is the promise `img`/`vid` already made about model, size and seed.
  One live set and one dormant, swapped in `setKind` — the invalid state is
  unreachable because there is only ever one to read. What still differs beyond
  that is the options, which rebuild from `VIDEO_MODELS` — see below.

  **The split has happened, and the stash is now half of what it was.** The
  video side lost its prompt box to the scene composer: its prose lives in
  `scene.shots`, which belongs to one kind and is therefore not stashed at all —
  the same arrangement `regions` has. What still swaps is the negative, the pills
  and the chips. Two buffers behind one live slot was the right *shape* for the
  transition and it is still the right shape for what is left of it.
- **Copy is a last resort — but a number is not a value it can show.** Design
  first, then an icon, then words: "Krea 2 Turbo", "16:9", "720p", "5s" name
  themselves. Twice the icon was not enough and the design changed instead of a
  caption being added: keyframe tiles mark where the frame sits in the clip, and
  a tile that appears replaced the checkbox that used to reveal it.

  The hyperparameters were where this rule was pushed past what it can carry.
  "32" is a rank, an alpha, an epoch count or a seed with equal plausibility,
  and the icons standing in for those words failed the only test that matters:
  someone who has trained these models for five years had to hover every
  numeric field to find out what it was. An icon is a rebus for a word you
  already know — it cannot tell you *which* hyperparameter you are looking at.
  So every numeric field carries its name, and the tooltip is promoted from
  repeating that name to saying what the number does. The rule survives, with
  its scope corrected: a control that shows its own value gets no label, and a
  bare number is not a value.

- **A trigger phrase is text. A LoRA is a file, and it is a chip.** This section
  used to defend `<lora:name:0.8>` in the prompt — Automatic1111's syntax, the
  notation anyone who has trained these models already types — against a row per
  LoRA that cost 56px plus a wrapped select, 380px of canvas for four filenames
  and eight digits. The row objection was right and is not what changed.

  **What changed is that the two things in that field were never one thing.** A
  trigger phrase reaches the encoder, so it belongs in the prompt, positioned
  where the sentence wants it. A LoRA never reaches the encoder at all —
  `stripLoras` deleted it from the string before `/api/generate` was called and
  the stack travelled in its own field. It was inline for exactly one reason: so
  a number could be typed beside it.

  **And the argument that kept it there does not hold for the main prompt.** The
  claim was that a row *could not say where in the sentence the LoRA applies* —
  true in a region, false on the canvas, and the since-deleted `useDocument` said
  so itself: *a token's position in the main prompt means nothing to the backend,
  which reads them into a stack.* So the canvas paid for a parser, a
  caret-targeting scheme and a drag subsystem to buy a property only a box has,
  and what the position means in a box is *which box*, which a control living on
  that box says without any syntax.

  So a chip: the name, then a circle carrying the strength, with the second value
  — text-encoder weight on image, and on video nothing, because H3's stack is
  one number — disclosed on a click, because it is omitted far more often than
  not. They fold behind
  `▸ 4 LoRAs`, a disclosure built as one more `#shot-peek`: zero pixels at rest,
  in flow so nothing sits on top of a render, and it does not close on scroll,
  which `Popover` does and which disqualifies it for a box you type numbers into.
  The count is a word rather than a pip, for the reason the regions button
  learned. `+ LoRA` stays exactly what it was — a picker in the strip — and the
  division is the shot rail's: the door adds, the disclosure shows.

  **A region has a dropdown on its own card**, one per box because that is the
  node's shape, and picking a second replaces rather than being refused. That is
  what retired caret-targeting: `+ LoRA` writing into "whichever of the two
  fields you last had the caret in" was answering *where* with a guess about
  where you were looking.

  **The picker writes nothing.** No token, no strength, and no trigger phrase —
  it shows the known phrase as text to read and place yourself. Three of the
  note's lines went with the syntax and two of them went because they became
  *impossible*: a chip is picked from a list, so no name resolves to nothing and
  none resolves to two. The third, a LoRA whose phrase is missing from the prose,
  can still happen and is deliberately unsaid. The failure it named is real —
  the weight loads, the render changes a little, and it reads as a LoRA that did
  nothing — so this is a decision to manage it by hand, not a discovery that the
  warning was wrong. `/api/state` still serves `trigger_word` per entry, which is
  where a check or an agent reads the fact.

  **`<lora:…>` left in a prompt is now text.** Nothing parses it, converts it or
  migrates it. `stripLoras` survives on the send path alone so a reused prompt
  carrying one does not render the literal word "lora" — it removes text on the
  way out rather than reinterpreting it in the box.

  **Dragging a LoRA is gone; dragging a file is not.** `lora/drag.ts` and its
  private MIME type were there to answer *where* — a click inherited the caret, a
  drag named its own target. Once every target owns a control there is no *where*
  left, and the gesture was long: open a menu at the prompt bar, at the bottom of
  the screen, then haul up to a box on the canvas. A reference image comes from
  the Finder, where a drag is the only gesture there is, so every file drop is
  untouched. See `docs/design-notes/loras-are-not-text.md`.

  **Everything above about *placement* is the Krea 2 side, and only that.** The
  `▸ 4 LoRAs` disclosure, `+ LoRA` in the strip, the region dropdown, `LoraBox`,
  `LoraButton` and every id in `check_loras.py` describe a console with a prompt
  box in it. **The video side does not have one**, so none of them are there: a
  LoRA is a chip on the one rail, beside the cast and the shot pills, and the
  picker is a mark on the field's trailing edge. Do not port these controls
  across, and do not read a rule about `#lora-box` as a rule about video.

  What *is* shared is the idea and nothing else: **a LoRA is a file plugged into a
  module, a trigger phrase is text, and the two do not live in the same place.**
  That is the whole transferable claim. Where the chip sits, what opens it and
  what the count says are per-side, because the two sides no longer share a
  console at all.

- **A region is drawn on the canvas, and so is everything about it.** The boxes
  are the list: ⌘-drag or double-click on the frame to place one, drag it to
  move it, drag any of its eight handles to size it. Click one and it *opens* —
  a card rooted in the box's own near edge, holding its sentence, its LoRA
  strength, its photograph and its four coordinates.

  **A click is not a drag, and the card is what the difference buys.** Selection
  used to be the open state, and that put a 296px panel over the picture for
  every gesture that touched a box — including the ones that were not about what
  is inside it. Framing is a run of those: draw, move, reshape, draw again. Worse
  than sitting there, it *ate presses*, because the layer refuses anything inside
  `.rins`: a handle or a whole box lying under the card was not adjustable at
  all, and the frame's card parks in the bottom-left corner of the picture
  whenever nothing is selected. So the same press on the same rectangle means
  two things and the *release* says which — it stayed put, so show me this one;
  it travelled, so move this one. A press anywhere that is not the card puts the
  card away, which is one rule rather than a list of places, and it holds off the
  layer too: the console, the strip and the page are outside the card as much as
  bare canvas is.

  **That dismissal is what took drawing off the plain drag.** A card you put away
  by clicking outside it, on a surface where most of the canvas *is* outside every
  box, cannot also be a surface that draws on a plain press — every dismissal
  would leave a rectangle behind. Both gestures that make a region are now
  deliberate: ⌘, which already meant "a new one, here" where there is no bare
  canvas left to aim at, and a double-click on bare canvas. A ⌘-click that never
  travels and a double-click land on the same 0.28 × 0.6 rectangle, so the two
  ways in do not produce two house styles.

  **Editing what is inside a box and redrawing the box are two different acts,
  and one control for both taxed the frequent one.** Changing a region's
  sentence or its photograph is what you do a dozen times an hour; moving the
  rectangle is rare, and mostly happens once before the first render. A sentence
  is not geometry, so the frequent act needs no rectangles on screen at all —
  which is what `store.edit` encodes, in three states:

  - `off`, the moment any render lands. Nothing is drawn. The boxes are still
    there, still masking their LoRAs, still sent with the next run; they are
    *addressable rather than drawn*. Hover names what you would be touching, a
    plain click opens it, and that is the whole entry — there is no chip and no
    glyph, because a control that reveals the boxes is a control competing with
    the picture it would reveal them on.
  - `content`, the frequent act: that one box's card, with only its own hairline
    for scope. No other rectangles, and **no four coordinates** — they are the
    escape hatch for the *rectangle*, so putting them over a render you are
    judging is the other scope leaking back in. It does carry that box's
    handles, and it is worth reading why that is not the same leak.
  - `geometry`, the rare one: every box, its handles, the snapping, the four
    coordinates and the frame's own card. Asked for with **⌘-click**, or a long
    press where there is no modifier to hold.

  **An open box is adjustable, and the gate was never about that box.** This
  read `no handles` for a version, on the argument above. True of the rectangles
  you have *not* touched, and the whole force of it comes from there — what
  geometry cost was eight boxes and sixty-four handles arriving over a render you
  were judging. Nothing is drawn until you click, so the box whose
  card is open is not one of those; it is already the scope of a card, already
  carrying a hairline, and already the thing you are looking at. Being told to
  press again with a modifier held to move *that* rectangle is friction with
  nothing behind it — the same invented friction as gating geometry behind
  clearing the canvas, arrived at from the other direction. So the open box
  behaves exactly as it does in geometry: handles, drag to move, snapping. Every
  other box stays undrawn and stays a tap.

  The coordinates stay behind the gate anyway. Dragging is the gesture, they are
  the escape hatch from it, and a render you are judging is the wrong place for
  four numeric fields — which is the original sentence, unweakened, because it
  was about the numbers rather than about the rectangle.

  **⌘ means geometry, and it means it in both places.** ⌘-drag on the frame was
  already "a new box, here", so this is the existing meaning extended rather
  than a second rule. It gates the mode and nothing else: one press that both
  revealed the boxes and drew one would answer "show me the boxes" by adding a
  ninth to the eight it just showed you. Gating geometry behind *clearing the
  canvas* was the version before this, and it was invented friction — a step
  that exists to protect a render nobody was being asked to give up.

  The two-press rule holds *while there is something behind the gate*. With no
  boxes at all, "show me the boxes" shows nothing, so the reveal-only press was
  a dead press — and worse, the layer used to mount over a render only when a
  box already existed, so on a fresh session ⌘-drag over the picture hit bare
  pixels and did nothing at all: the only way to draw a first box was to clear
  the render you wanted to draw against. The layer now mounts over every
  render (it paints nothing in `off`, so an empty one costs no chrome), and a
  ⌘-drag with an empty set falls through the reveal and draws the first box in
  one gesture.

  The four coordinates were the right parameter and the wrong primary. "0.5 0
  0.5 1" is a rectangle you rebuild in your head every time; the rectangle is
  not. They survive in the card, where they are the escape hatch and, during a
  drag, a readout that moves — dragging teaches the numbers, and the numbers
  never taught the dragging.

  **The canvas has no layers, so the boxes do not have them either — and they
  had the worst kind, the kind nobody chose.** They are absolutely-positioned
  siblings in `regions` order, so the DOM supplied a z-order out of an array
  index and it decided every click, every hover and every drop: whichever box
  was drawn *last* won. A performer placed inside a wide background box could be
  reached and the background box could not, and the eight handles of anything
  underneath were simply unreachable. Nothing in this feature has ever had a
  front or a back — a box is an area of one picture — so the ordering was a
  storage detail the picture could see.

  The rule that replaces it is one this file already states, in the Phase 6
  list, for exactly this ambiguity: **resolve toward the smaller object.**
  Widening a selection is cheap and obvious; guessing large silently edits the
  wrong scope. `boxAt` takes the smallest box containing the point, and a handle
  — smaller than any box — comes first, but only where it is *drawn*. An
  invisible handle is not an object, and letting one win would give every box
  four edges of theft over its neighbours at a target nobody can see.

  **Hover had to move with it, and that is the part that looks optional.** CSS
  `:hover` follows paint order, which is the ordering just deleted, so the
  hairline saying "this is what you would be touching" would have named a
  different box than the click opened. It is `.rbox.under`, set by the same hit
  test — one answer, painted and acted on. Not `.hot`: that is already this
  page's word for a drag held over a drop target, and `check_drop.py` reads it
  as exactly that.

  A drop reads the same test, which is the case with no undo. It used to take
  the topmost box while the caption named the topmost box too — consistent, and
  both wrong the moment a small box sat inside a large one.

  Two arrangements preceded this and each fixed the last one's cost. A row per
  region was a sentence and four numbers that needed a 32px picture of those
  numbers beside it to be legible at all, and at ~74px each the eight boxes the
  backend allows came to ~592px of console against a 54dvh cap: the feature's
  fullest state broke the rule the console exists to hold. One shared inspector
  row fixed the height and left the other half — you dragged a rectangle at the
  top of the screen and said who was in it at the bottom, and the row could not
  say which of the two scopes in it you were looking at. The card costs the
  console nothing at all: `probe_console.py` measures arming at 0px, where the
  row cost 44.

  **A card is not a portal.** Every other floating thing here is `Popover`,
  positioned in viewport pixels, which is why it closes on scroll — right for a
  menu and absurd for the field you are typing into. The card is a child of the
  region layer, so it is a percentage of the same box the rectangles are a
  percentage of: it moves when they move and is clipped by the frame, which is
  the correct answer rather than a limitation. It also has three placements
  rather than two — under the box, over it, and *inside* it pinned to its own
  bottom edge, because the two rectangles arming seeds are full-height columns
  and the very first card anyone sees is the one with nowhere outside the box to
  go.

  It stays mounted through a drag and goes transparent instead. Unmounting was
  the first version and it re-ran the card's mount effects on release, which
  pulled focus into the prompt after *moving* a box — so ⌫ edited text instead
  of deleting the rectangle, which is exactly the keyboard fault this redesign
  set out to fix, reintroduced by the fix. `check_regions.py` found it on its
  first run, which is the argument for that file existing.

  Boxes snap to halves, thirds and quarters and to each other, which is what
  makes an even split a gesture rather than a menu; Alt suppresses it. Arming
  the mode seeds two half-width columns rather than explaining anything, because
  two rectangles appearing on the canvas is the instruction. What the boxes are
  drawn *on* is the frame at the render's aspect, or the scene plate if one is
  dropped, or the last render — adjusting boxes against the picture you actually
  got is the reason they are still there afterwards.

- **The frame is a place too, and it has the same card.** Scene, outfit and the
  region weight are about every box at once, so they cannot live on any one
  rectangle. They used to sit in the same row as the selected box's own fields,
  separated by a rule and by nothing else — two scopes in one strip, with the
  reader left to infer which was which. Now the scope *is* where the card is,
  and it is reached by one button in the corner of the layer, because a tap on
  bare canvas is already taken: it draws a rectangle. Escape is the keyboard
  half of the same thing. There is always exactly one card and it is about
  whatever is selected.

  The frame's card is also the only thing on screen that can say which engine
  the run takes — a plate moves the render onto a krea2edit compose that
  regenerates the whole frame and is several times slower, and no arrangement of
  rectangles shows that.

- **The map went with the row.** A 52px SVG of the boxes lived beside the
  inspector because the boxes come off the picture the moment a render lands,
  and something had to lead back to them. Two things did, and it was the lesser:
  the mode button reveals rather than disarms on the first press after a render,
  and a file dragged over the window brings them back on its own. What the map
  additionally did — reach a box that is overlapped or off-screen — is Tab,
  which cycles them and always could, once clicking one actually focused it.

  **`Nothing sits on top of a render` survives, and the three states above are
  what it costs to keep it.** It is written here because a canvas-native rework
  is exactly the moment someone assumes it was relaxed. A finished render puts
  the boxes away and the card with them, every time, including after the render
  you asked for while editing them — `off` is re-entered on every land, so the
  clean picture is never something you have to restore. A persistent hairline
  was considered and rejected for the same reason it always was: chrome on the
  one surface the layout exists to keep clear.

  What did change is the second half of the sentence. Nothing is *drawn* on a
  render; the regions are still *there*, and touching one opens it. That is not
  a relaxation, it is Phase 6's own rule arriving early — *every element
  addressable at all times, none of them drawn as a control* — and it is the
  first place in this app where the two halves of that rule are separable at
  all. The distinction to hold on to: this rule is about paint, and it never had
  anything to say about reach.

- **What a place carries is attachments, and an attachment has a role.** A
  photograph in a box is that character's likeness; the two frame-scope plates
  are the scene the picture is generated inside and an outfit transferred onto
  the subjects. They reach the backend under three different names — a region's
  `ref`, `scene`, `outfit` — and they are one thing: a picture, and what it is
  for. So the page holds one record for all three, and one function attaches it,
  taking the place as an argument. That argument is the entire difference
  between "this character" and "this scene".

  This is the axis the next capability arrives on, and it is worth writing down
  before there is anything at stake. **There is no ControlNet in this backend** —
  no node, no mask primitive, nothing in the catalogue — so this is a question
  about shape rather than a feature to build, which is the cheap time to answer
  it. The shape is already latent in three places and named in none: the region
  mold's V12 parameters are `ref_strength`, `ref_start_percent`,
  `ref_end_percent` and `ref_feather` scoped to a box, which is ControlNet's
  parameter set exactly and is hardcoded; `refs_json` is already a positional
  list of `{role, note}`, pinned to scene and object; and the video side's
  `SHOT_REF_ROLES` is the same vocabulary in the open.

  So a ControlNet is `depth`, `pose` or `edges` — a role on a picture, not a
  feature. V12 returns `(MODEL, +COND, −COND)`, so a `ControlNetApplyAdvanced`
  slots between it and the `KSampler` without touching anything above it, and
  adding one should be four things: a catalogue entry, which gets download UI
  and status for free; a preprocessor on a **CPU** container; a branch in
  `_krea2_graph`; and a row in the role table. **Zero new front-end surface** —
  dropped on the frame it is frame-wide, dropped on a box it is masked.

  **The standing veto: no conditioning panel, no layers list, no "control"
  tab.** When four roles exist, a list of them somewhere is the obvious cheap
  fix and it is precisely the month-four failure Phase 6 names. An attachment is
  drawn on the thing it is attached to, or it is not drawn. And `Derived or
  invented, always visible` lands here too: a depth map is derived, so it has to
  be visible on the attachment and cheap to reroll.

- **Every picture the model can be given sits in one row, and the two halves
  dim each other.** Keyframes and references are the same decision made two
  ways — they load different transformers, so one excludes the other — and they
  used to be two rows: keyframes parked at the right of the strip among the
  numeric controls, references in their own row below. Two pairs of unlabelled
  36px dashed tiles, one row apart, telling each other apart by tooltip.

  What that cost was not tidiness. The keyframe tiles were never found at all,
  and dropping photos into the reference tray *looked* like filling keyframe
  slots that kept growing — which is exactly what the tray does and exactly what
  a fixed pair must never look like. Side by side with a rule between them, the
  tray that grows and the two slots that do not are told apart by shape, which
  is the thing a tooltip could not do. Whichever half is out of play goes dim
  rather than disappearing: the row's job is to show that these are
  alternatives, and a control that vanishes when you fill its neighbour teaches
  nothing except that the page lost it. References win when both are attached,
  because that is what the run does, so they are the half that stays live.

  The note under the field says "keyframes are ignored" only when there is a
  keyframe to ignore. It said it unconditionally, which made the page's one
  mention of keyframes a warning about something you did not have, pointing at
  a control this layout had already made unfindable.

- **A box takes a photograph as well as a LoRA.** `regions_json.ref_image` is a
  latent mold: it pulls that rectangle toward that face during sampling, stacks
  with the box's LoRA, and runs on the fast single-pass path. It is not an
  `extra_ref_*` plate, so — unlike the scene and outfit tiles — it needs no
  identity-edit weight and never moves the run onto the slow krea2edit compose.
  A box with a photo and no LoRA is a character with no training run behind it,
  which is worth having on a platform whose other half is a trainer.

  Photos are capped at 1536px on the long side before they are encoded, because
  eight of them in one JSON body is the payload this invites. That is
  deliberately the *only* cap: the node's own `ref_max_side` is set to 0 so the
  resizing happens in one place and the two cannot end up fighting over which
  one shrank the picture. The job record carries a bool, never the bytes — it is
  polled every 400ms.

- **Grey is whose, the underline is reach — and the mirror had to change layers
  to say the first one.** The prompt box marks what the interpreter supplied,
  inline in the sentence, and that is two channels rather than one. Every
  *element* carries a dotted underline, because an element is a thing the
  document can act on: one is a single gesture from being something else.
  Whether the words in it are the person's or the model's is a separate claim
  laid over the top, and only the model's are grey.

  An earlier note here said derived text carries no mark at all, on the grounds
  that underlining what somebody wrote would mark nearly every word. That was
  right about a rule which underlined all derived text and wrong about this one:
  an element is an anchor and what hangs off it, so the marks land on the
  handful of things in the sentence there is anything to *do* to, and the
  connective tissue between them carries nothing.

  What it cost is worth recording, because it looks like a regression on the way
  past. The mirror used to paint decoration onto transparent glyphs while the
  textarea supplied the visible text, and a *colour* is impossible that way
  round — the half that can be styled is the half nobody can see. So the ink
  moved one layer down: the mirror's glyphs are what you read and the textarea
  is transparent with `caret-color` set. It keeps the caret, the selection, the
  undo stack and every chord in `keys()`, which is the whole reason this is
  still a textarea rather than a contenteditable.

  **The only thing a grey run does that plain text cannot is reroll**, and that
  is the whole of its affordance: rooted at the run's own end, revealed while
  the caret is inside it, gone when it leaves. Editing one needs no control at
  all — `remap` drops the mark the edit landed on, so the words turn dark and
  become yours with no gesture and nothing to commit. An inline editable rooted
  in the run was considered and rejected: a second text surface competing with
  the one underneath it, buying nothing typing does not already do.

  A reroll lands three ways — new text, the same text, a rejection — and **the
  in-flight state sits on the affordance and settles identically for all
  three.** Not a pulse under the words: that is motion on the render surface to
  announce a null result, and a flicker that fired for *identical* while a
  rejection stayed silent would build a channel telling you which way the
  validator went, which is the one thing the silent degrade exists not to say.

- **A prompt is written by reordering it.** ⌥← / ⌥→ moves the clause under the
  caret one slot along, because "in soft window light" belongs before the
  subject as often as after and doing that by hand is a select, a cut, a click
  and a paste — four gestures, each with its own way of eating a comma. The
  separators are slots and they do not move: the commas and line breaks stay
  where they are and the text between them changes places, so a prompt written
  across two lines still has two lines however many times you press the chord.

- **The empty prompt box is the worst control on the page, so on the video side
  there is not one.** H3 does not read a paragraph; it reads a document with
  named fields, published in the model repo. The composer offered a textarea for
  it, and every symptom of that is the same symptom: there is no slot for camera
  direction, so every position is a guess; tone and genre belong to a clause
  with no name on screen; the place a reference image's description belongs is
  not on the page, so it goes in the only box there is. A documented grammar
  presented as free prose reads as superstition — whether a comma or "the woman"
  versus "a woman" changes the take is not something anyone can infer — and
  every guess is paid for at the length of a take.

  The palette below was the first half of the answer and the **scene composer**
  is the second — see "The scene composer" further down, and
  `docs/design-notes/console-ladder.html` for the design it was built to.

  So the closed vocabulary is a **palette**: a door in the strip beside `+ LoRA`,
  a popover of small animated tiles, and the pills themselves under the prompt.
  The prompt field keeps only what nothing else can say — who is in the shot and
  what happens. This is the label rule above applied to words instead of
  numbers, and it is the one place on the page where an icon can teach: a tile
  *shows* a dolly-out, which is the thing neither the word nor a static picture
  does.

  **The door was a wordless glyph, and that — not its room — was the fault.** It
  was a 34px mark beside `+ LoRA`, in that same scan-when-you-know strip, next to
  a second opaque mark doing the same disappearing act for regions. Regions left
  that row for good, because it is a canvas verb. Shot did
  not: it writes into the prompt, and `+ LoRA` is already precedent for the strip
  hosting a door that does that. What it needed was a word.

  It carries the word "Shot", and the icon rule is what says so rather than what
  it breaks: a glyph can carry a control whose home you are already in, and it
  cannot announce a *destination* — and eighty-seven tiles behind one press is a
  destination. The tile beside the word is the teaser, and it **animates on hover
  and is frozen otherwise**, which is the pills' own rule: the page at rest runs
  nothing, so a loop under the prompt would be motion competing with the canvas
  for attention you have not asked it for. Hover is the asking. The glyph follows
  the kind — a dolly-out on video, a framing on images — because the camera group
  is video-only and teasing a move the model cannot make is a promise the run
  will not keep.

  **It sat at the head of the pill rail for a version, and the measurement is what
  moved it back.** The rail is the right *room* — the door and the words that come
  out of it in one place, needing no caption — and it was the wrong price. A row
  that exists only to hold one button costs 34px of a console capped at 30% of the
  viewport, at rest, forever: resting went 120px to 154px, while the rail carrying
  sixteen pills only ever added 29px on top of that, because then the row is
  holding something. `#shot-rail:empty{display:none}` is back, and the rule it
  encodes — the rail costs nothing until there is something in it — turns out to be
  load-bearing rather than tidy. A line per button is not a price this console can
  pay, and the general form is worth keeping: **a row is affordable when it carries
  content, and never when it carries one control.**

  Three rules hold the rest together:

  - **No pills, no document.** With nothing chosen the compiler returns the
    typed text byte-for-byte. Every prompt written before this still means what
    it meant, and the document only appears once you have said something that
    needs one.
  - **The compiler never rewrites your sentence.** It closes it if you did not,
    and it chooses the separator in front of it — a leading clause's full stop
    softens to a comma before a lowercase fragment, so "A medium close-up, a
    portrait of k3nan." rather than a capital that would silently turn a `k3nan`
    trigger word into `K3nan`. That is the whole extent of it. What is inside
    `<d>…</d>` is not touched at all, which is the guide's own rule and the one
    place ordinary tidying would corrupt the output invisibly.
  - **`non_diegetic_music: N/A` is the default, and is worth the feature on its
    own.** H3 invented a soundtrack for every clip because nothing had ever told
    it not to.

  `/api/compile` is the same compiler on the same CPU container, so the
  disclosure under the rail shows the exact document that would run. A preview
  with its own implementation is a preview that can disagree with the run, which
  is worse than none; and without it the only way to answer "where did my camera
  direction go" was to render again.

- **A reference chip carries what it is *for*.** Identity, wardrobe, location,
  style, prop or action, from the chip's own menu, compiling to the guide's
  `<Subject 1> is the person in <Picture 1>` and a matching retention line.
  This is what makes "do not describe the picture you attached" enforceable
  rather than advice: there is now somewhere for that description to go which is
  not the prompt field, and it is one click. Roleless chips run exactly as they
  did.

- **The ratio picker and the pixel boxes are one control.** There is only ever
  a width and a height; the ratios are shortcuts to a pair of them. Picking one
  writes the boxes, typing in the boxes selects Custom, and Custom is the only
  option that spells out its own size, because it is the only one with no name
  that implies it. Sizes snap to 8 on the way out — the pipeline floors to the
  VAE's grid regardless, and a box that keeps 1000 while the model renders 992
  is a box that lied to you. It was 16 for a while, on the theory that the DiT's
  patch of 2 needs an even latent. It does not: `SingleStreamDiT.forward` pads
  up to the patch size and crops the result back, so an odd latent costs one row
  of patches. Snapping to the coarser grid was not protecting anything — it was
  refusing every second size the model can render.

  **A tier is a multiplier, except where it is a family.** 1K, 1.5K and 2K are
  one trained set — ~1.03 MP on a 64 grid — multiplied, because scaling a bucket
  keeps the shape the model knows while deriving one from the ratio at a new edge
  length lands on a size nothing was trained on. 1.3K is not that set made bigger:
  it is ~1.62 MP on a 32 grid, and 1.25x the bucket agrees with it on 1:1 and
  misses every other row by 32-48px — 1440x1120 where the family's 4:3 is
  1472x1120. So that tier pins its eight sizes and the multiplier is what a bucket
  falls back to when it has no pin, which is what keeps adding a ninth ratio one
  line rather than a column somebody has to remember to fill in.

  The ratio set is one vocabulary across every tier, not a set that changes under
  the scale buttons — so 21:9 and 4:5 exist at 1K too, and theirs are the only two
  rows in the table that are derived rather than inherited. Derived to the label
  *exactly*, 1568x672 being 7:3 and 896x1120 being 4:5, because where the model was
  never given a bucket there is nothing to be faithful to except the name.

  The swap arrow between the two boxes belongs to the same control, which is
  what decides where it lands: transposing a preset re-selects the preset for
  the transposed pair rather than dropping to Custom. Five of the eight transpose
  within the menu and three do not, so 4:3, 21:9 and 4:5 **deselect** — Custom, at
  the transposed pixels.

  That is the answer and not the cost of one. 3:4 was on the menu purely to give
  this button somewhere to land, and it went with the ratio set; the flip of a
  ratio the menu has no counterpart for *is* a custom ratio, so a tile left lit
  there would be the picker claiming a shape it does not offer. Re-adding a ratio
  to keep a tile lit is the tail wagging the dog, and special-casing it in
  `swapSize` is worse — it lights a tile for a bucket the picker cannot otherwise
  reach. Arrow keys are part of it too: ↑/↓ steps
  a numeric box by one and ⌘↑/⌘↓ by eight, which on Width and Height is the
  VAE's grid, so the coarse step always lands on a size the model can render.
  Nothing is snapped there, because typing 1153 already shows 1153 until you
  leave the field — an arrow is a faster way to type a number, not a second way
  to commit one. Fields whose useful range is 1.0 to 1.4 carry their own
  `data-step`; a shift of 1.15 stepped by 8 leaves behind every value the model
  accepts.

  **That paragraph described an intent the page did not deliver for most of its
  life, and the note is kept because the shape of the fault recurs.** In
  `UI_HTML` the handler was delegated from three sections and the sizer popover
  was appended to `<body>`, so with it open there were two pairs of boxes —
  `#g-w`/`#g-h`, inside a listening section and invisible, and `#sz-w`/`#sz-h`,
  visible and outside every one of them. The arrows stepped the inputs nobody
  could focus and did nothing to the boxes you actually type in, and nobody
  noticed for months, because a rule that reads as shipped is one nobody thinks
  to test.

  The React front end does not inherit it: the component owns its own keys, so
  where a popover renders stops being able to break them.
  `tools/ui-checks/probe_size.py` asserts it and now passes every row — it used
  to fail two by design, and that exemption went with `UI_HTML`.

`tools/preview_ui.py` serves the built bundle against stubbed JSON, so the front
end is worked on locally instead of paying an image build and a cold start per
CSS change. Its stubs are shaped to hold the awkward states — a missing model, an
uncaptioned dataset, a prompt too long to belong in a gallery card.

**A stub that omits a menu is a preview of a control that does not exist**, and
it fails silently — the case that taught this was `rewrite_ops` missing from
`/api/state`, which rendered no Enhance button at all, so the one surface that
feature had was invisible in the very file that exists to make the front end
developable without a GPU. The button is gone now; the rule is not. Every menu
the page builds itself out of is pulled from app.py rather than transcribed, and
pulling one means pulling **what it references** — a subset naming a constant
nobody pulled raises `NameError` from inside app.py, which reads as a broken
preview rather than an incomplete pull. That is `_from_app.py`'s one failure
mode and the reason its subset is named rather than pattern-matched. It fired
twice in one session over `_stage_*`, which is the rule working.

## Where the console redesign got to

**Promote and demote — done.** The rule that settled it is under "The page":
sort by how often you reach for a control, not by whether it is per-take or
per-session. What you touch constantly is the row; what you touch rarely is
behind the model button. Measured, the image strip was 1016px
of controls and the video 979px, most of it spent on things a take does not
change. Seed and the batch count went into the Sampling popover — a seed is
*reused off a result*, so the gesture happens after a render and not before, and
a batch is a run parameter like steps. The GPU went under the gear, because it
is chosen once a session and already confirms a cold start when it changes. The
video side finally got the size control the image side had, collapsing a
separate aspect select and tier select into one button reading `16:9 · 768p`.
Result: 732px and 652px.

Two more have left the strip since, and not to a popover: `#g-shot` and
`#g-regional` were not controls at all. See the icon rule under "The page" —
each went to the surface it acts on, which leaves this row holding the size, the
LoRA picker and the model button, and nothing that has to introduce itself.

**Unbounded buttons — done.** Ten controls each in their own box is ten boxes
competing with the picture above them, and the chrome is not what makes a
control a control: the value is, and every one of these already shows its own.
So the box is spent only when it is doing something — the pointer is over it, or
a popover is open from it. Scoped to the console: Train keeps its edges, because
it is a form you fill in rather than a row you scan, and a form without field
edges is a worse form. The line that decides is the same one that governs
labels — a control whose value *is* its label can lose its box; a free-text
field cannot, because an empty one has nothing to show and no edge to aim at.

**Bar or rail — settled: the bar stays.** The dead-space table under "The page"
says a rail would fit on desktop, and it is still not the answer. A rail is a
dated shape that will read as of-its-decade long before this app stops being
useful, and the overlay that would have avoided both was built and rejected for
putting chrome on the render. What the measurements really argued for was less
in the bar, not the bar somewhere else — which is what the popovers did.

The direction instead is **unbounded buttons**: controls that carry no chrome
until they need it, so the strip stops looking like a strip. It is a step toward
the console disappearing entirely, which is the actual goal the rail was a
detour around.

## The scene composer

**The video side has no prompt box.** H3 reads a document with named fields —
shots, cut times, speaker IDs, a retention line per subject — and a text field
has no shots in it, which is why `_compile_h3_prompt` could not emit half the
grammar it was written against: the composer never collected it. The design is
`docs/design-notes/the-scene-is-the-prompt.md` for the argument and
**`docs/design-notes/console-ladder.html` for the layout**, and the second
supersedes `scene-composer.html`, which hides the canvas behind a 132px stub. The
ladder draws all five states at true proportion against `CONSOLE_BUDGET`, and its
own footnote is why to trust the picture over the prose: twice while it was being
drawn the two disagreed and the prose was the optimistic one both times.

**The degrade is exact, and everything rests on it.** One shot, no cast, no
pills, and `readScene` returns null: no `scene` key is sent and the run is the
typed text byte-for-byte, in a box that is identical to the prompt box it
replaced, under the same `#prompt` id. Nothing about the video side changes until
somebody asks for a second shot. That is what let this land in three commits
rather than one, and it is the rule to check first if any of it is ever moved.

- **The gesture creates the cast.** Typing `@` floats a picker off the caret;
  picking makes the member and drops the handle. So the cast has no empty state,
  because it is never on screen before it exists — which is `+ LoRA` inverted:
  there you open a picker in order to insert a name, here the name you are
  already typing *is* the picker. The three doors on the field's trailing edge
  reach the same thing without knowing the gesture.

  **Where the caret is comes from the mirror**, and that is the third job it has
  done. It is a glyph-for-glyph copy of the textarea, so a zero-width span
  spliced into it at the caret's index sits exactly where the caret does — the
  only way anything on this page can answer that question. It was built for
  provenance marks, kept when they were deleted on the grounds that unwinding it
  bought nothing, and it is now load-bearing again.

- **The share is a readout, not an input.** A shot's slice of the clip is the
  length of what you wrote about it. That is not a shortcut — it is what removes
  a 9px drag target instead of enlarging it to a thumb-sized one, which is the
  question a phone frame asked and the console could not afford to answer. Extent
  drives prominence; `beats` survives as the precision escape hatch, the way a
  region's four coordinates do.

- **Rows divide the field's existing allowance rather than adding to it.**
  `growRows` is `autoGrow` over n boxes and reduces to it exactly at n=1, so a
  four-shot scene costs what a long prompt costs and the 30% budget needs no
  second arithmetic. Measured at 1512x982 with three shots and the source row
  showing: 287px, 29.2%.

- **The file is the slot, and a form field is a closed vocabulary.** A character
  had five labelled squares — Face, Wardrobe, Body, Voice, Motion — on the rule
  that *the slot is the role*, and the rule was true and the surface was wrong.
  Five squares means exactly five sayable things about a person: *"the coat she
  is wearing in the other photo"* and *"her posture from the one on the stairs"*
  were **structurally unsayable**, and H3's conditioner is an LLM text encoder
  that parses grammar, so the way to say them is to say them. That is the *Prose,
  not tags* rule in the root file, which was written about captions and is a UI
  rule.

  So one well takes anything the member can hold and the media picks the slot —
  a picture is what they look like, a recording is what they sound like. See
  `slotFor`. Anything else you want referenced is its own named thing, which is
  what the guide means by a subject: *"visible content abstracted from reference
  assets"*, not only people.

  **Their face is what the rail draws.** The chip carried a 5px dot, filled or
  hollow, as the one fact it could hold about somebody — a person reduced to a
  monogram on a token, in a product whose strongest lever is that it can be
  handed a photograph. The dot survives only where it was always right: a member
  with a description and no picture, whom `_h3_label` compiles to prose rather
  than to a `<Subject N>`.

  **And the name is what you write with.** `Maya` is something you can remember
  and build an action around; `<Subject 1>` is not. Both numberings are assigned
  behind the surface — `<Picture N>` from upload position, `<Subject N>` from the
  named thing — and neither is ever shown.

  What survives from the five squares is the one rule worth keeping: a file the
  member cannot take does not highlight. The rejection is the absence of an
  invitation, which arrives before the drop rather than after it; read off
  `dataTransfer.items` during the drag, the one moment the answer is available
  and still useful. `DropTile` alerts instead and is right to — its tiles all
  take images, so a wrong file there is worth naming.

  **This made voice cloning reachable, and it had been built and invisible.**
  `/api/video` already accepted `ref_audios`, staged them as `.wav` and fed them
  to the graph; `H3_AUDIO_NOUN` already emitted `<Audio 1> is the voice-timbre
  reference for <Subject 1>` as its own line with its own retention marker.
  Nothing in the page had ever let anyone reach it. Two drops on one target now
  do.

- **Templated is ours; intent is theirs — and a reference's *type* is what
  swaps a sentence between the two.** The division the whole video side runs
  on, stated once: instruction text the format dictates — the alignment
  sentences, the retention grammar, the voice-timbre line, a character sheet's
  citation, `subject_definitions`' shape — is the compiler's to write, and the
  person never sees or types it. Everything carrying intent — what a subject
  *is*, what a picture *provides*, the prose, the dialogue — is theirs and
  travels verbatim. When a control would ask the person to type instruction
  text, the control is wrong; when the compiler would rewrite intent, the
  compiler is wrong.

  A character sheet is the worked example. Marking a picture as a sheet is a
  *typed fact*, one click on its row — and zero clicks when Cast made the
  sheet, because provenance is certain. Marked, the compiler writes the
  guide-shaped citation (the sheet's views and on-image labels doing the
  defining, MiniMax's own construction for a labeled reference card) and the
  row's note field becomes a grey readout saying so — derived, always visible,
  never a field asking for words the run will not read. The description stays
  the person's and leads the definition untouched.

- **One rail, three kinds of chip** — cast, shot pills, LoRAs. They were three
  mechanisms in two places and they are one tool: each is something attached to
  the thing you are making, and each opens its own card when touched.

- **The pool is keyed by what travels**, not by the file on disk. A photograph
  and its own re-export shrink to the same bytes, and two entries in
  `references[]` pointing at one picture would put every `<Picture N>` after the
  first off by one with nothing on screen saying so. It also makes the numbering
  derivable rather than kept in step by hand, which is what `refs`/`refRoles`
  had to do — not fixed, *unrepresentable*.

- **Everything that floats is portalled and fixed.** `.console` is
  `overflow:auto` — the rule that stops a full console pushing the canvas out of
  frame — so a child of it floating above its own top edge is clipped by it. Not
  `Popover` either: that closes on scroll, which is right for a menu and wrong
  for a box you type a name into. Two faults found here and both are the same
  shape: a ref read during the parent's render is null and nothing re-renders to
  correct it, so the menu placed itself at 0,0; and picking left the menu open on
  the name just chosen, which is a picker that will not take yes for an answer.

### The compiled document is one gesture away, and it is not a developer tool

**Reading the prompt is not a step in composing.** It was frame 6 of the ladder
and that was the error: the document is secondary — hidden most of the time,
one gesture away.

**And "secondary" is a claim about attention, never about aesthetics.** The
devtools metaphor below got taken literally for a while — monospace, an
uppercase COMPILED tag, a view-source costume — and the owner's correction is
the record: *"my only point was that it should be treated as something that
doesn't need to be seen most of the time but is one click away. I did not
actually want it to look like code. My whole design thesis is it should not
feel utilitarian."* The document is prose the model reads, so it is set as
quiet prose in the app's own type, at a reading measure. What survives of the
metaphor is the **access pattern** only:

- **No button says "inspect".** A chord and a context menu, never advertised.
  ⌘⌥U, or right-click the console — one of the two has to be findable without
  being told. It stands aside for a text field's spelling menu or an image's
  Save Image rather than deciding it outranks the platform.
- **It is not in the console budget, because it is not in the console.** It takes
  the canvas the way devtools takes the window, and nobody is judging a render
  while reading a prompt. The surface that kept breaking the budget stops
  competing for it.
- **A textarea, not a `<pre>`,** and the edit is what runs. Unlike an
  inspector's edit, this one cannot be allowed to evaporate — that would be a
  render you cannot reproduce. So editing **detaches** — one bit for the whole document, visible in
  the header, one gesture back. Not per-field pinning: six independent states
  nobody asked for, and an attempt to make a derived surface partly
  authoritative. Either the scene is driving it or you have taken it over, and
  you can always see which.
- **It shows derivation.** Put the caret in a `[Shot N]` block and that row lights
  up below. Only while attached — a detached document is arbitrary text and its
  markers are claims about nothing.

`prompt_compiled` is its own key on `/api/video` rather than a rewrite of
`prompt`, and this is the case that separates the two halves of a run:
`prompt_typed` stays the prose somebody wrote and only the receipt is overridden.
Folded together, the sidecar's intent field would hold a six-field schema and
Reuse would load it into the first shot's row and compile *that* on the next run.
It is stripped and never `_oneline`d — that function exists because a newline
inside a *field* ends it early, and this is the document, where one field per
line is the format.

- **The caps are published, binding and were completely invisible.** Nine images,
  three videos, three audio — and **twelve across all types**, which is the one
  that actually binds and the one nothing ever said. Nine photographs plus three
  voices is exactly twelve, so a fully cast scene has no room for a reference
  video; three audio caps how many voices one generation can clone.

  `/api/video` refuses past twelve with a named error before renting anything, so
  the note is not protecting the run — it is protecting the *discovery*. Finding
  out at Generate, after casting nine people, is finding out too late to be told
  cheaply, which is the keyframe note's argument exactly.

  It says nothing until something is wrong, like `loraNote`, and it lives in the
  reserved 18px row so Generate does not move under a hand already reaching for
  it. The per-type sentence comes first because it names the *silent* half:
  `/api/video` slices `references` to nine **before** it sums the twelve, so
  twelve photographs and three voices passes the total check with three pictures
  quietly gone, and the refusal then arrives as a cast member pointing past what
  was uploaded — a true error about the wrong thing.

  **And it counts what will travel.** `VideoNote` read `refs`/`refVids` alone,
  correct while those were the only way to attach a picture and blind from the
  moment the composer arrived: nine photographs on nine cast members counted as
  zero. `refBudget` follows `videoBody`'s rule — the cast's files when there is a
  cast, the flat trays otherwise, never both.

  Finding it also caught this file's one named failure mode in the act:
  `max_ref_audios` was served by app.py and missing from `preview_ui.py`'s state
  stub, so the front end would have developed against a limit that did not exist.

### A scene is longer than a generation

H3 tops out at `H3_MAX_FRAMES` — 345 frames at 24fps, about 14.4 seconds — so
anything with more than one beat in it is several runs. Film granularity is
*frame · shot · scene · sequence · act · film*; this platform is for the first
three, and past a scene you are in an NLE and out of scope.

**The model gives you frames and shots. The scene is ours, and it is not a
capability.** `[Shot N]` with cut times already lives inside one generation. What
turns several generations into one scene is that the cast, their photographs,
their voices, the look and the LoRAs never belonged to a take in the first place
— so the only thing asked of H3 on the next run is the same characters again,
plus whoever is new. Chaining is context the person should never have to rebuild,
not something the model has to learn.

`Continue` on the canvas is the whole gesture. It reads the last frame out of the
clip that just landed, hands it to `first_frame`, and clears the prose. Three
things about that are worth stating:

- **The last frame is read in the page, not on a GPU.** The bytes are already
  served to a `<video>` on the canvas, so a route that decoded a frame
  server-side would be a GPU-adjacent container doing work a decoder in the page
  does for free. See `lastFrame` — `seekable.end` rather than `duration`, because
  a fragmented MP4 reports `Infinity` for the latter and seeking to it hangs.
- **It is best-effort.** A codec the browser will not decode yields no frame and
  the next take opens cold rather than refusing to start. Continuity is the
  point, not a precondition.
- **`last_frame` is cleared on the way.** A first and a last together are
  `fl2va`, which would pin the new take's *ending* to the old take's ending —
  the exact opposite of continuing.

What does not carry is the prose. A take is a beat, and reopening on the sentence
you already rendered invites editing the last one rather than writing the next.

`takes` lives on the store and deliberately **not** inside `scene`: that type
mirrors `_validate_scene` and is the request body, so a record of what has
already rendered would be results in the payload.

### What it does not do yet

- **Blocking is not reachable from it, and that is now a decision rather than a
  gap.** `_stage_*` and `_stage_boxes` stay live in the compiler with nothing
  filling them in, and `web/src/blocking/` stays a throwaway probe. A ground
  plane cannot express camera roll, a body that changes pose mid-shot, light as
  a transition, or a location change inside one unbroken take — every one of
  which a sentence carries for free. The exchange rate settles it: learn a camera
  rig, gain five prepositional phrases.
- **None of it has been measured against a render.** It has been driven and read.
  `tools/prompt_ab.py` is the measurement that is not a proxy and it has not run.

## What this page may not grow

The end state these serve is in `docs/roadmap.md`, together with the half of the
argument that is about prompts and authorship rather than about surfaces. This
half is here because a veto that is not in context when someone adds a panel is
not a veto.

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
