import { useEffect, useLayoutEffect, useRef, useState } from 'react'
import { createPortal } from 'react-dom'

import { useSettled } from '../ui/gesture'
import { useStore } from '../store'
import { save } from './arsenal'
import { Material } from './Material'
import { RETENTION, RETENTION_LABEL, handleOf, type CastMember } from './model'

/**
 * What a cast member is made of, disclosed only for the chip you touched.
 *
 * **It costs the console nothing.** A card rooted on its chip and floating over
 * the canvas, the way a region's card is rooted on its box — which is what makes
 * the depth affordable at all: five slots per member and eight members is a panel,
 * and a panel is the month-four failure Phase 6 names.
 *
 * **And it is a form, deliberately.** Labels, field edges, one row per thing you
 * can say about somebody. The first version stripped all of that on the "copy is
 * a last resort" instinct and it was the wrong rule to reach for: that one governs
 * a row you *scan*, and this is a thing you *fill in* — the same distinction that
 * keeps Train's edges while the console's controls lose theirs. A form without
 * field edges is a worse form. Unlabelled, it read as a request for five files.
 *
 * **Not a `Popover`, and a portal anyway.** Popover closes on scroll, which is
 * right for a menu and wrong for a box you type a name into — the objection
 * `LoraBox` records. But the console is `overflow:auto`, so a child of it floating
 * above its own top edge is clipped by the very rule that stops a full console
 * pushing the canvas out of frame. So: portalled and fixed, placed off the chip's
 * own rect, and closing on nothing but a press outside it or Escape.
 */
export function CastCard({ member }: { member: CastMember }) {
  // 'idle' → 'saving' → 'saved', falling back to idle after a beat. A save is a
  // network write; a button that gives no sign it worked teaches people to
  // press it three times.
  const [kept, setKept] = useState<'idle' | 'saving' | 'saved'>('idle')
  const s = useStore()
  const box = useRef<HTMLDivElement>(null)
  const name = useRef<HTMLInputElement>(null)
  const [at, setAt] = useState<{ left: number; top: number } | null>(null)
  // **Nothing on this card answers the gesture that made it.** It mounts at the
  // cursor — the mention menu picks on mousedown, because the textarea has focus
  // and a click would blur it first — so without this the mouseup of that same
  // press landed on whichever slot was now underneath and opened a file dialog
  // for Motion, with the card nobody had seen yet behind it. `pointer-events`
  // rather than a guard per control, because the ✕ is under the same cursor and
  // that one deletes the thing you just made.
  const settled = useSettled()

  // **It finds its own chip**, rather than being handed a coordinate by the rail.
  // The rail is a masked scroller, so a card rendered inside it would be clipped
  // by the very overflow that keeps nine chips one row tall — it has to be a child
  // of the console, and then the only thing that knows where the chip is is the
  // DOM. Re-measured on every render for the reason `Popover.place` is: the rail
  // reflows as chips are added, and a card that keeps its old left is one pointing
  // at whatever moved into that spot.
  useLayoutEffect(() => {
    const el = box.current
    const chip = document.querySelector<HTMLElement>(`.chip[data-cast="${member.id}"]`)
    if (!el || !chip) return
    const r = chip.getBoundingClientRect()
    const w = el.offsetWidth
    const h = el.offsetHeight
    const left = Math.max(8, Math.min(r.left, window.innerWidth - w - 8))
    // Above the chip, always. Below it is the console and then the bottom of the
    // window; there is never room, and a card that flipped down would be a card
    // half off the screen.
    const top = Math.max(8, r.top - h - 8)
    // Only when it moved: a fresh number every render is a state change every
    // render, which is the "Maximum update depth exceeded" that unmounts the whole
    // app rather than just this card.
    setAt((cur) => (cur && cur.left === left && cur.top === top ? cur : { left, top }))
  })

  // Capture-phase mousedown, which is the pattern in this app that holds: focusout
  // covers only the case where focus lands somewhere that takes it, and a click on
  // the canvas or a dead area of the bar moves focus nowhere at all.
  useEffect(() => {
    const down = (e: MouseEvent) => {
      const t = e.target as Element | null
      if (!box.current?.contains(t) && !t?.closest?.('.chip.cast')) s.setRailOpen(null)
    }
    const key = (e: KeyboardEvent) => { if (e.key === 'Escape') s.setRailOpen(null) }
    document.addEventListener('mousedown', down, true)
    document.addEventListener('keydown', key)
    return () => {
      document.removeEventListener('mousedown', down, true)
      document.removeEventListener('keydown', key)
    }
  }, [s])

  // A member made by typing `@` already has its handle; one made from the rail's
  // own door does not, and the name is the only thing it cannot do without.
  useEffect(() => {
    if (!member.name) name.current?.focus()
  }, [member.name])

  return createPortal(
    <div className="tcard" ref={box}
         style={{ left: at?.left ?? 0, top: at?.top ?? 0,
                  visibility: at ? 'visible' : 'hidden',
                  pointerEvents: settled ? undefined : 'none' }}>
      <button type="button" className="x" title={`Remove @${member.name || member.kind}`}
              onClick={() => { s.dropCast(member.id) }}>×</button>

      {/* The name first, then what it is, then what you have of it. The face
          led here for one version, which was right while `Material` was a single
          well and wrong the moment it became a list — a stack of references is
          not a headshot to sit a name beside. */}
      {/* The `@` is inside the field's edge because it is part of the value, not
          a label on it: this string *is* the handle a shot mentions it by, and
          renaming rewrites it across every row as a visible find-and-replace.
          **The name is what you write with** — `Maya` is something you can
          remember and build an action around, `<Subject 1>` is not, so the
          numbering is assigned behind this and never surfaces. */}
      <div className="tname-row">
        <span className="at">@</span>
        <input id={`cast-name-${member.id}`} ref={name} className="tname"
               value={member.name} spellCheck={false} placeholder="name"
               onChange={(e) => { s.patchCast(member.id, { name: handleOf(e.target.value) }) }} />
      </div>

      <label htmlFor={`cast-note-${member.id}`}>Description</label>
      {/* **Who they are comes before what you have of them.** A cast member with a
          sentence and no photograph is a perfectly good subject — `_h3_label` falls
          back to exactly this. With a picture it trails the definition,
          `<Subject 1> is the person in <Picture 1>, mid-thirties`; with none it is
          the whole of what the encoder is told, and the alternative is the literal
          word "sam". */}
      <input id={`cast-note-${member.id}`} className="tnote" value={member.note}
             spellCheck={false}
             // One placeholder, because there is one kind — and the guide's own
             // example is exactly this shape: "<Subject 1> is *the young woman*
             // in <Picture 1>". Whether it is a person, a room or a coat is
             // something this sentence says, not something a menu asked first.
             placeholder="what it is — the young woman, a rain-slick alley, an olive coat"
             onChange={(e) => { s.patchCast(member.id, { note: e.target.value }) }} />

      <label>References</label>
      <Material member={member} />

      {/* **Save is the Arsenal's whole write surface, and it is deliberate.**
          "It never remembers unless told" — this button is the telling. The
          character lands at characters/{name}/ on the volume as plain files
          plus a note.txt, readable in a terminal with no app, and comes back
          by typing @name in any future session. Only once there is a name and
          something attached: a character with no files is a name, and a name
          is already in your head. */}
      {handleOf(member.name) !== '' && member.refs.length > 0 && (
        <button type="button" className="tkeep" disabled={kept !== 'idle'}
                title={`Save @${member.name} to the arsenal — recall them by typing @${member.name} in any session. Saving again replaces.`}
                onClick={() => {
                  setKept('saving')
                  void save(member).then((err) => {
                    if (err) { alert(err); setKept('idle'); return }
                    setKept('saved')
                    window.setTimeout(() => { setKept('idle') }, 1600)
                  })
                }}>
          {kept === 'saving' ? 'Saving…' : kept === 'saved' ? 'Saved ✓' : 'Save to arsenal'}
        </button>
      )}

      {/* **Only once there is something to retain.** This sat on the card from
          the moment a name existed — a control asking how much of a reference
          has to survive, shown to somebody who has attached no reference. That
          is the empty state this file's own rule forbids: nothing exists until
          you make it, and a member with no picture has nothing for the marker
          to be about. It compiles to nothing either, since `retention_analysis`
          only ever names a subject. Found by opening the card and reading it
          as a first-timer rather than by reading the compiler. */}
      {member.refs.length > 0 && <>
      <label htmlFor={`cast-hold-${member.id}`}>Retain</label>
      {/* How much of the reference has to survive into the render — the marker in
          `retention_analysis`. It reached the document at the strictest value with
          nothing setting it, which is right for a face and wrong for a location you
          want referenced rather than reproduced. */}
      <select id={`cast-hold-${member.id}`} className="thold" value={member.retention}
              onChange={(e) => { s.patchCast(member.id, { retention: e.target.value }) }}>
        {RETENTION.map((r) => (
          <option key={r} value={r} title={r}>{RETENTION_LABEL[r]}</option>
        ))}
      </select>
      </>}
    </div>,
    document.body,
  )
}
