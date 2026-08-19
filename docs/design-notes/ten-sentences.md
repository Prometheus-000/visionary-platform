# The Eleven Sentences

A veto list, not a manifesto.

These are not aspirations. Each one exists to kill a specific thing when the cheap fix gets proposed. When someone — including you, in month four — suggests adding a panel, a toggle, a settings screen, or a confirm button, one of these sentences should be the reason it doesn't ship.

If a sentence has never vetoed anything, it isn't earning its place. Cut it.

---

## 1. If a gesture cannot communicate intent without a UI label, the answer is not a control panel — it's a better spatial gesture or a smarter model interpretation.

Every other sentence on this list says *don't*. This one says what to do instead, which is why it sits first: the rest are instances of it. When something won't fit, the failure is in the gesture or in the interpretation. It is never in the absence of a panel.

**Kills:** the panel, obviously — but also the tutorial. Onboarding, coach marks, first-run overlays, the little animated hand demonstrating a swipe. Those are labels wearing a costume. If a gesture needs teaching, the gesture is wrong.

**On iPad:** the two escape hatches are spatial and semantic. Either the gesture becomes more physical — bigger, more direct, more obviously about the thing it acts on — or the model gets better at reading what the touch meant. Both are real work. That's the point: this sentence exists to make sure the work gets done rather than deferred into chrome.

---

## 2. The canvas never changes.

Character, wardrobe, action, location, cinematography — same white surface, every time. What changes is what the surface is willing to hear.

**Kills:** tabs, rooms, navigation, mode screens, a spine, an org chart, a "back" button.

**On iPad:** there is one view. No split view, no popovers that own the screen, no drill-down. If you can swipe between screens, it's already wrong.

---

## 3. What you touch decides the mode.

You don't enter character mode. You touch her face. You don't enter blocking mode. You touch the floor. Mode is a consequence of attention, never a precondition for it.

**Kills:** tool palettes, mode switchers, a selected-tool state, anything you must do *before* the thing you want to do.

**On iPad:** ambiguity resolves toward the smaller object — her hand over the counter is her, not the room. Widening is a cheap, obvious gesture. Guessing large silently changes the wrong scope.

---

## 4. Everything is live. Nothing is labeled.

Every element is addressable at all times — her, her coat, the counter, the light, the camera — while nothing is drawn as a control. This is the hardest engineering in the product, and it exists so the screen can stay empty.

**Kills:** inspectors, sidebars, property panels, toolbars, persistent chrome of any kind.

**On iPad:** no hover means no tooltips and no reveal-on-hover. Discoverability is by touch or it doesn't exist. Every affordance must survive a finger arriving with no warning.

---

## 5. Nothing asks for confirmation.

Move her and the frame changes. There is no apply, no generate, no regenerate, no commit.

**Kills:** modals, confirmation dialogs, "are you sure," staged edits, a render button.

**On iPad:** undo is a two-finger gesture and it is always available. Reversibility replaces confirmation entirely — that's the trade, and it only works if undo is genuinely total.

---

## 6. Duration starts at zero.

A still is the default. Time is something you add. Video is a feature, not a path.

**Kills:** a timeline on first run, a beat strip anyone has to see, video-first onboarding, any flow that treats a photograph as the degenerate case.

**On iPad:** someone who wants one image should finish and leave without ever learning that motion exists.

---

## 7. The frame is computed, never authored.

Blocking produces the frame. Move a mark and the shot changes. You never compose a frame by dragging its contents.

**Kills:** a viewport where you drag people to make a picture, camera settings divorced from staging, any control that fakes causality.

**On iPad:** this is what makes direct manipulation honest — you're moving the world, not the image of it.

---

## 8. The camera is a character.

It has a mark, a path, a want, and a body with real limits. It can be late. It can look away. It can be wrong.

**Kills:** the camera as a settings group, lens choice as taste, "camera controls."

**On iPad:** the camera is touchable like anyone else, and it's cast like anyone else.

---

## 9. Derived or invented, always visible.

Anything read from your words is marked one way. Anything filled in for you is marked another, and is cheap to reroll. This is the entire trust surface and it requires no dialogue.

**Kills:** a chat panel, an assistant sidebar, opaque proposals, clarifying questions. Every question asked is a small failure — pick something, mark it invented, move on.

**On iPad:** the mark must be legible at arm's length, on a glossy screen, in daylight.

---

## 10. The arsenal is closed until you reach for it.

A drawer, not a workspace. Browsed and selected, not searched. Entities are **applied, never imported** — edits in the scene are scene-local, edits in the library propagate. Two different acts, two different places.

**Kills:** a persistent library rail, a docked browser, an asset manager, anything that gives storage permanent screen space.

**On iPad:** an edge gesture opens it, taking something closes it. It never lingers.

---

## 11. The blank canvas never remembers unless you tell it to.

The library grows. The compiler doesn't change. Same words, same result, a year later. The system never notices your patterns and never proposes them.

**Kills:** personalization, suggestion engines, learned preferences, recently-used defaults, anything that pre-populates a new scene.

**On iPad:** the first screen of the thousandth session is identical to the first screen of the first.

---

# What touch forces

Designing for iPad isn't a platform choice, it's a constraint that does the work for you.

- **No hover.** Every progressive-disclosure trick dies. Either it responds to touch or it isn't there.
- **No right-click.** No context menus. No hidden second layer of function.
- **No keyboard.** Every shortcut I'd have reached for — lock, deselect, duplicate — has to become a gesture or a direct manipulation, or be cut.
- **Imprecise input.** Small controls are impossible, so you manipulate objects instead of widgets. This is the constraint that most directly produces the design you want.
- **Pencil pressure and duration.** A brush has no settings; it has behavior under pressure. That's literal here, and it's the standard: controls are learned by pressing harder and watching, not by reading what they're called.
- **One thing at a time.** No windows, no panels floating over other panels. The platform enforces the discipline.

---

# What the trackpad threatens

Touch gives you the discipline for free. The trackpad takes it back, and every affordance it hands you is a way to violate sentence 1 while feeling productive.

**One interaction model, not two.** The pointer is a finger with better aim. Anything that only works with a cursor is a fork, and forks rot — the touch version becomes the degraded one within two releases. Design to the coarser input and let precision be a bonus.

**Hover is the seduction.** It's the single biggest threat on this list, because it makes tooltips, reveal-on-hover controls, and floating toolbars feel free. They aren't free — they're labels, and they don't exist on iPad, so anything that depends on them is a feature half your users can't reach.

Hover is permitted for exactly one thing: **showing what you would be touching.** Her, not the room. The coat, not her. That's spatial feedback about scope, not a control being revealed, and it actually solves something iPad can't — on touch, ambiguity resolves silently toward the smaller object and you find out after the fact. On trackpad you find out before. Same rule, better feedback.

**The cursor is the only label-free indicator you get.** It can say what a drag will do without drawing anything. Use it — it's the desktop answer to "how do I know what I'm about to work on," and it costs no pixels.

**Precision invites small controls. Refuse.** The trackpad can hit a 12px target. Building one means the iPad build needs a different layout, and you've forked. Objects get manipulated directly at the size they appear.

**No pressure, so nothing may require it.** Pencil pressure is an accelerator on iPad, never a requirement, or the web build loses a capability outright.

**Keyboard accelerates, never enables.** A shortcut is an invisible label — it has to be taught, which sentence 1 kills. Every shortcut must have a direct-manipulation equivalent that a first-time user finds by reaching. Undo is the one exception, because it's universal muscle memory rather than something you learn here.

**Bigger screen means more white, not more UI.** The web build's extra space is not an opportunity. It's a test.

---

# MVP scope

**In:**

- One entry surface: describe, or drop references, or both, or neither
- Whatever arrives produces committed entities and a standing world — output shape constant, only the derived-to-invented ratio changes
- Still frames only
- Touch-to-edit, mode following touch
- The arsenal: promote, browse, apply
- Promotion available from anywhere — a still, a frame, a plate. The best version of a thing shows up while you're doing something else.

**Out — deliberately:**

- Motion, duration, beats, timelines
- Blocking floor plan, camera paths
- Coverage, shot lists, chaining
- Collaboration, export pipelines, project management

The MVP is: **wake up, describe a fragment, get a world you can touch, keep what worked.** If that loop is clean and empty, the rest is addition. If it isn't, nothing downstream saves it.

---

# The failure to guard against

Not the model. Not latency. Not scope.

It's month four, when something doesn't fit cleanly and the cheapest fix is a panel. That's the moment this becomes a node graph with better typography.

Read the list before you take the cheap fix.
