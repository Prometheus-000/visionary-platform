import { useEffect, useRef, useState } from 'react'

import { IconArrange, IconPhoto, IconTrash } from '../icons'
import { Menu } from '../ui/Menu'
import { NumInput } from '../ui/NumInput'
import { usePopover } from '../ui/Popover'
import { DropTile } from '../media/DropTile'
import { shrinkB64 } from '../media/files'
import { caretProps, dropCaret } from '../lora/caret'
import { regionNote } from '../lora/note'
import { chipFrom, loraIndex } from '../lora/tokens'
import { hydrateRegion, listCharacters, saveRegion,
         type SavedCharacter } from '../scene/arsenal'
import { handleOf } from '../scene/model'
import { attached, regionsLive, useStore, type EditMode, type Region } from '../store'
import { takeResumeFocus } from './focus'
import { clamp01, distribute, readRegions } from './geometry'

/**
 * The box, opened.
 *
 * A rectangle on screen can be 60px wide and what it holds needs about 280, so the
 * card cannot be the box — but it can be *rooted* in it, hanging off the box's near
 * edge and sharing its stroke, which is the difference between the object answering
 * and a panel arriving next to the object. That distinction is the whole reason this
 * replaced a row in the console rather than being moved into one somewhere else.
 *
 * **A child of the layer, not a portal.** Every other floating thing on this page is
 * `Popover`, and this deliberately is not one. A portal to `<body>` is positioned in
 * viewport pixels, so it drifts the moment the canvas scrolls — which is why Popover
 * closes on scroll, a behaviour that is right for a menu and absurd for the thing you
 * are typing into. Living inside `#region-layer` makes the card a percentage of the
 * same box the rectangles are a percentage of: it moves when they move, scrolls when
 * they scroll, and is clipped by the frame, which is the correct answer rather than a
 * limitation — the card belongs to the picture.
 *
 * **Selection is the open state.** There is no toggle and nothing to dismiss: the
 * selected box is the one being edited, and the one being edited is the one showing a
 * card. It goes transparent mid-drag, because a card over the rectangle you are
 * dragging is a card in the way of the thing it describes.
 *
 * Transparent rather than unmounted, and that distinction cost a real bug. Returning
 * null while `boxDrag` was set tore the card down and built it again around every
 * gesture, so its mount effects re-ran on release — which pulled focus into the prompt
 * field after *moving* a box, not just after drawing one. ⌫ then edited text instead of
 * deleting the rectangle, which is precisely the keyboard fault this redesign set out
 * to fix, reintroduced by the thing that fixed it. Found by `check_regions.py` on its
 * first run, which is the argument for the file.
 */
export function Inspector({ mode }: { mode: EditMode }) {
  const s = useStore()
  if (!regionsLive(s) || mode === 'off') return null
  const r = s.regions[s.rsel]
  // The frame's card is a geometry-scope thing — the scene, the outfit and the weight
  // every box is multiplied by are decisions about the whole arrangement, not about the
  // one performer you touched. So `content` shows a box or it shows nothing.
  if (!r) return mode === 'geometry' && s.cardOpen ? <FrameCard drag={s.boxDrag} /> : null
  return (
    <>
      {/* Open, not selected. See `cardOpen` in store.ts: the card arriving on every
          gesture that touched a box put a 296px panel over the picture for the whole
          of a framing session — and the layer refuses presses inside `.rins`, so
          whatever lay under it was not adjustable at all. */}
      {/* **Keyed by the box, so selecting another one is a remount.** Everything
          in this card is per-box state that assumes exactly that — the caret
          sink's withdrawal ("this one goes on every render of selection, not
          just on unmount"), the resume-focus flag, the Arsenal listing and the
          save button's `Saved ✓`. It held only by accident: pressing another box
          puts the card away first, so `cardOpen` flickered false and unmounted
          it. Tab cycles selection without that press, and there the same
          instance was reused — a `Saved ✓` sitting on the next character's card,
          and a recall list fetched for a box you left. */}
      {s.cardOpen && <BoxCard key={r.id} r={r} i={s.rsel} drag={s.boxDrag} mode={mode} />}
      {/* The card's numbers, for the length of the gesture that takes the card away —
          and now for every drag, because a drag is what closes the card in the first
          place. Gated on geometry, which is the gate the numbers themselves carry: a
          mode with no coordinates in it does not grow some for a drag. */}
      {s.boxDrag && mode === 'geometry' && <Readout r={r} />}
    </>
  )
}

/**
 * Where the card hangs off its box, in the layer's own percentage space.
 *
 * No measurement, and that is the point: the rule is expressed in the same 0..1 the
 * boxes already live in, so it needs neither the layer's pixel size nor a
 * ResizeObserver, and it cannot disagree with where the rectangle was painted.
 *
 * Three placements, not two. Under the box when there is room under it, over the box
 * when there is room over it — and *inside* it, pinned to its own bottom edge, when
 * there is neither. The third is not an edge case: the two rectangles this mode seeds
 * are full-height columns, so `y + h` is 1 and the very first card anyone sees is the
 * one with nowhere outside the box to go. Without it the card placed above the frame,
 * where `overflow:hidden` ate it, and the feature looked like it had not been built.
 *
 * Horizontally it hangs off whichever vertical edge of the box is nearer the middle,
 * so the card always grows inward and `max-width` keeps it inside the picture.
 */
function anchorTo(r: Region): React.CSSProperties {
  const fromLeft = r.x + r.w / 2 < 0.5
  const right = clamp01(1 - (r.x + r.w)) * 100
  const side = fromLeft ? { left: `${clamp01(r.x) * 100}%` } : { right: `${right}%` }
  if (r.y + r.h < 0.62) return { ...side, top: `calc(${clamp01(r.y + r.h) * 100}% + 7px)` }
  if (r.y > 0.38) return { ...side, bottom: `calc(${clamp01(1 - r.y) * 100}% + 7px)` }
  // Pinned inside the box, at the bottom — which is the one placement that shares a
  // corner with the frame button. `max` floors the offset rather than capping the
  // width, because a right-hand full-height box puts the card at `right: 0%` and a
  // narrower card would still have started on top of the button.
  return {
    ...side,
    ...(fromLeft ? {} : { right: `max(${right}%, 44px)` }),
    bottom: `calc(${clamp01(1 - (r.y + r.h)) * 100}% + 7px)`,
  }
}

function BoxCard(
  { r, i, drag, mode }: { r: Region; i: number; drag: boolean; mode: EditMode },
) {
  const s = useStore()
  const field = useRef<HTMLInputElement>(null)
  const index = loraIndex(s.state)
  const pick = usePopover()

  // The caret sink is a module-level ref holding one element, so the field has to
  // withdraw when it goes — and this one goes on every change of selection, not just
  // on unmount. A stale element would keep taking `+ LoRA`'s writes and paint them
  // nowhere.
  useEffect(() => {
    const el = field.current
    return () => { if (el) dropCaret(el) }
  }, [])

  // The other half of the iteration loop. A render remounts this card, so a ⌘Enter
  // fired from inside the sentence would otherwise land you on a new picture with the
  // caret gone. `focus.ts` carries one bit across that gap: if the generate came from
  // a region field, put the caret back so the next tweak is a keystroke, not a click.
  // Guarded by the flag so it never steals focus from a card you merely selected —
  // which is what the "nothing focuses anything here" note above relies on.
  useEffect(() => {
    if (takeResumeFocus()) requestAnimationFrame(() => field.current?.focus())
  }, [])

  // Nothing focuses anything here. A new box arrives with its caret in the prompt and
  // an existing one arrives with focus on the rectangle, and only the drag that ended
  // knows which just happened — so `RegionLayer` decides it, the way the vanilla page's
  // `selectRegion(idx, focusPrompt)` did. Guessing it from the box's contents looks
  // equivalent and is not: an *empty* box you clicked to delete would take the caret
  // and swallow the ⌫.

  const num = (key: 'x' | 'y' | 'w' | 'h', label: string, title: string) => (
    <div className="opt n" data-lb={label}>
      <span className="lead">{label}</span>
      <NumInput value={fmt(r[key])} fine={0.01} bigStep={0.1} title={title}
                onValue={(v) => {
                  const n = parseFloat(v)
                  if (Number.isFinite(n)) s.patchRegion(i, { [key]: clamp01(n) })
                }} />
    </div>
  )

  return (
    <div className={`rins${drag ? ' dragging' : ''}`} id="region-inspector" style={anchorTo(r)}>
      <CharacterRow r={r} i={i} />

      {/* The sentence gets the row to itself. Beside the photo tile it measured 168px
          of a 296px card, which is about four words of a field whose placeholder is
          eight — and what a box holds is mostly this. */}
      <div className="rins-row">
        {/* Direction for one performer, not a second scene description. The split is
            the only rule this feature has: the prompt in the console is what every
            performer is standing in, this is what this one is doing, and the box
            already said where.

            Words only. This used to take the same `<lora:…>` syntax as the main
            field, on the argument that one notation at two scopes needs nothing to
            explain which is which — true, and it made the field do two jobs. The
            LoRA is the dropdown below, which says *which box* by being on it.

            "e.g." because the example was read as content: a screenshot arrived
            with "the box is missing Scene | Outfit" and the placeholder quoted
            back as the user's own sentence. And the tooltip names the other two
            surfaces, because this field is where someone who cannot find them is
            already looking — the card can announce a capability where an icon
            cannot. */}
        <div className="opt wide">
          <input id="r-prompt" ref={field} value={r.prompt}
                 placeholder="e.g. a man in a denim jacket, laughing"
                 title="This performer only — who they are and what they are doing. Their face is the Photo or LoRA below; a scene or outfit photo for the whole frame goes on the frame card, behind the corner button. Where they stand is the box — do not write a position here, it is already said."
                 onChange={(e) => s.patchRegion(i, { prompt: e.target.value })}
                 {...caretProps('region', (v) => useStore.getState().patchRegion(
                   useStore.getState().rsel, { prompt: v }))} />
        </div>
      </div>

      <div className="rins-row">
        <DropTile id="r-ref" label="Photo" value={attached(r, 'identity')} glyph={<IconPhoto />}
                  title="Character reference — a photo of this person. Pulls the box toward that likeness during sampling; stacks with the LoRA, and works without one. Not for clothing or places — an outfit or scene photo goes on the frame card."
                  onFile={async (f) => {
                    const b64 = await shrinkB64(f)
                    if (b64) s.attach(i, 'identity', b64)
                  }}
                  onClear={() => s.attach(i, 'identity', null)} />

        {/* One per box — the node's own shape, so a dropdown rather than an
            add-many, and picking a second *replaces* rather than being refused:
            choosing a different name for a box that has one is "this character
            instead", and answering that with an error would send you somewhere
            else to fix it.

            `opt ib`, which is the same treatment every other door on this page
            wears, and it is here because the class it used to carry — `pickish` —
            was declared in this file and defined in no stylesheet. A button with
            no rules gets the UA's own: a white fill and an outset border, inside
            a card that is `#101010`. That was the whole of it visually, and it
            cost more than a look: there is no `.opt:disabled` either, so with an
            empty index the control was a dead white box indistinguishable from a
            live one, which is a click that does nothing and says nothing about
            why. Both halves are fixed below and in ui.css.

            `+ LoRA` rather than "No LoRA", for the same reason the strip's door
            says it: an empty control should name the act, not the absence. Word
            for word what `LoraButton` reads, so one label opens a LoRA picker
            wherever you are — and it is the *name* that is the value here, so the
            "a control that shows its own value gets no label" rule is satisfied
            the moment there is one to show. */}
        <button id="r-lora" type="button" data-lb="LoRA"
                className={`opt ib${pick.open ? ' on' : ''}${r.lora ? ' set' : ''}`}
                onClick={pick.toggle}
                disabled={!index.length}
                title={index.length
                  ? 'Which character this box is. One per box — the node takes one.'
                  : 'No LoRAs on the volume yet — train one under Train, or drop a '
                    + '.safetensors into loras/.'}>
          {r.lora ? r.lora.rel : '+ LoRA'}
        </button>
        {pick.open && (
          <Menu anchor={pick.anchor} onClose={pick.close}
                items={[
                  // Only when there is one to take off. Ticked-and-empty is a row
                  // that says "you have not chosen" to somebody who came here to
                  // choose, and it put the one inert row at the top of the list.
                  ...(r.lora ? [
                    { label: 'No LoRA', run: () => s.patchRegion(i, { lora: null }) },
                    { sep: true } as const,
                  ] : []),
                  ...index.map((l) => ({
                    label: l.token,
                    hint: l.trigger || undefined,
                    on: r.lora?.path === l.path,
                    run: () => s.patchRegion(i, { lora: chipFrom(l, true) }),
                  })),
                ]} />
        )}

        {/* Blank and disabled when the box holds no LoRA, because a strength with
            nothing to apply to is a number that lies. 1.3–1.4 is the node pack's
            own guidance for a character, which is what `fine` is sized to. */}
        <div className="opt n" data-lb="Strength">
          <span className="lead">Strength</span>
          <NumInput id="r-strength" value={r.lora ? String(r.lora.strength) : ''}
                    disabled={!r.lora} placeholder={r.lora ? '' : '—'}
                    fine={0.05} bigStep={0.25}
                    title="How hard this box's LoRA is applied. The node pack's guidance is 1.3–1.4 for a character."
                    onValue={(v) => {
                      const n = parseFloat(v)
                      if (Number.isFinite(n) && r.lora)
                        s.patchRegion(i, { lora: { ...r.lora, strength: n } })
                    }} />
        </div>
        <span className="grow" />
        <button className="opt ib" id="r-del" type="button"
                title="Remove this box — or select it and press ⌫"
                onClick={() => {
                  s.setRegions(s.regions.filter((_, n) => n !== i))
                  s.select(Math.min(i, s.regions.length - 2))
                }}>
          <IconTrash />
        </button>
      </div>

      {/* The escape hatch. Dragging is what makes "0.5 0 0.5 1" mean a rectangle; the
          numbers never taught the dragging, which is why they are the second row here
          and were the wrong primary in the row this replaced — and why `Readout` carries
          them through the one gesture that takes this card off the screen.

          Geometry only. They are the *rectangle*, and the card's other rows are what is
          inside it — which is the split this whole mode exists to make: touching a
          performer to change their sentence should not put four coordinates over the
          render, and it is the render you are trying to look at. */}
      {mode === 'geometry' && (
        <div className="rins-row nums">
          {num('x', 'X', 'Left edge, as a fraction of the width. 0 is the left of the canvas.')}
          {num('y', 'Y', 'Top edge, as a fraction of the height. 0 is the top of the canvas.')}
          {num('w', 'W', 'How much of the canvas width this box covers, 0 to 1.')}
          {num('h', 'H', 'How much of the canvas height this box covers, 0 to 1.')}
        </div>
      )}

      <KeepRow r={r} />
    </div>
  )
}

/**
 * Who this box is, and the only control on this card that outlives the picture.
 *
 * **The name is both halves of the Arsenal, which is why one field does it.**
 * Type a name nobody has saved and it is a name; type one you have and the row
 * under it offers them back, files and weight and sentence. That is the cast's
 * own rule — *the name you are already typing is the recall* — brought to the
 * side that never had it, rather than a Library button, a browse panel or an
 * import dialog, all of which the roadmap vetoes by name.
 *
 * It is above the sentence because it is the smaller, more definite fact: the
 * prompt is what they are *doing here*, and this is who is doing it.
 */
function CharacterRow({ r, i }: { r: Region; i: number }) {
  const s = useStore()
  const [saved, setSaved] = useState<SavedCharacter[]>([])
  const [open, setOpen] = useState(false)
  // Fetched when a card opens, which is popover speed: names, sentences and
  // filenames, never bytes. Same listing the mention menu takes.
  useEffect(() => { void listCharacters().then(setSaved) }, [])

  const q = handleOf(r.name)
  // A handle this box already *is* is not a thing to recall — the row would
  // offer to replace the character with itself. Everything else that matches
  // the prefix, and everything when there is no prefix yet, because an empty
  // field is the moment "who have I got" is the question.
  const hits = saved.filter((c) => c.handle && c.handle !== q
                                   && (!q || c.handle.startsWith(q)))

  return (
    <>
      <div className="rins-row">
        <div className="tname-row grow">
          <span className="at">@</span>
          <input id="r-name" className="tname" value={r.name} spellCheck={false}
                 placeholder="name — and how you get them back"
                 title="Who this box is. Naming them is what lets you keep them: save below, and typing the name in any future session offers them back with their photo, their sentence and their LoRA."
                 onFocus={() => { setOpen(true) }}
                 // On a timer, because the row below is a mousedown target and a
                 // blur fires first — closing the list on the way to clicking it
                 // is a picker that cannot be picked from, which is the fault the
                 // mention menu records from the other direction.
                 onBlur={() => { window.setTimeout(() => { setOpen(false) }, 120) }}
                 onChange={(e) => { s.patchRegion(i, { name: handleOf(e.target.value) }) }} />
        </div>
      </div>

      {open && hits.length > 0 && (
        <div className="rcast">
          {hits.slice(0, 6).map((c) => (
            <button key={c.handle} type="button"
                    // mousedown, not click: the field is focused and a click
                    // blurs it first, so the list is already gone by the time
                    // the press lands.
                    onMouseDown={(e) => {
                      e.preventDefault()
                      setOpen(false)
                      void hydrateRegion(r.id, c, loraIndex(useStore.getState().state))
                    }}>
              <b>@{c.handle}</b>
              <i>{c.note || (c.lora ? c.lora.rel : 'saved')}</i>
            </button>
          ))}
        </div>
      )}
    </>
  )
}

/**
 * The write surface, and the same one row the cast card carries.
 *
 * **Only once there is something to keep.** A named box with no photograph and
 * no weight is a name, which is already in your head — the sentence the route
 * refuses with, so the button does not offer an act the server will decline.
 * A LoRA counts: on a platform whose other half is a trainer, somebody who
 * exists as a weight and no reference photo is the case this is most for.
 */
function KeepRow({ r }: { r: Region }) {
  const [kept, setKept] = useState<'idle' | 'saving' | 'saved'>('idle')
  const name = handleOf(r.name)
  if (!name || !(attached(r, 'identity') || r.lora)) return null
  return (
    <button type="button" className="tkeep" id="r-keep" disabled={kept !== 'idle'}
            title={`Save @${name} to the arsenal — type the name in any future session to get them back. Saving again replaces.`}
            onClick={() => {
              setKept('saving')
              void saveRegion(r, name).then((err) => {
                if (err) { alert(err); setKept('idle'); return }
                setKept('saved')
                window.setTimeout(() => { setKept('idle') }, 1600)
              })
            }}>
      {kept === 'saving' ? 'Saving…' : kept === 'saved' ? 'Saved ✓' : 'Save to arsenal'}
    </button>
  )
}

/**
 * The four numbers, for the one gesture that hides the card holding them.
 *
 * `.rins.dragging` is opacity 0 for the length of a drag, and it is right — a card over
 * the rectangle you are dragging is a card in the way of the thing it describes — but it
 * took the coordinates with it, and three separate comments in this feature still said
 * the numbers are "a readout that moves while you drag". They were describing the console
 * row this card replaced, where the numbers sat below the canvas and could not be in the
 * way of anything. `RegionLayer` is still paying one store write per animation frame to
 * keep them moving, expressly so, into a surface at zero opacity. Dragging is what
 * teaches the numbers; a drag that shows none teaches nothing.
 *
 * Fixed in a corner rather than following the box, because a readout that moves with the
 * gesture is a second thing to track while you are already tracking the first — and
 * because staying put is what the row it is restoring did. Top-left is the corner nothing
 * else claims: `#canvas-acts` is top-right, the frame button is bottom-right, and
 * bottom-left is the frame card's, with a refused drop that sits there until the next
 * one.
 */
function Readout({ r }: { r: Region }) {
  return (
    <p className="rins-readout" id="region-readout">
      {([['x', 'X'], ['y', 'Y'], ['w', 'W'], ['h', 'H']] as const).map(([key, label]) => (
        // `fmt`, the same one the boxes print, so the readout and the card cannot
        // disagree about a rectangle they are both describing.
        <span key={key}><span className="lead">{label}</span>{fmt(r[key])}</span>
      ))}
    </p>
  )
}

/**
 * The frame is a place too, and this is what is attached to it.
 *
 * Scene, outfit and the region weight are render-scoped — they are about every box at
 * once, so they cannot live on any one rectangle. Before this they sat in the same row
 * as the selected box's own fields, separated by a rule and by nothing else, which put
 * two scopes in one control strip and left the reader to infer which was which. Here
 * the scope is where the card is: rooted in the frame rather than in a box.
 *
 * It is also the only thing on screen that can say which engine the run takes. A plate
 * moves the render onto a krea2edit compose that regenerates the whole frame and is
 * several times slower, and no arrangement of rectangles shows that.
 */
function FrameCard({ drag }: { drag: boolean }) {
  const s = useStore()
  const arrange = usePopover()
  const live = readRegions(loraIndex(s.state), s.regions, regionsLive(s)).length
  const note = regionNote(s, live)

  return (
    <div className={`rins frame-card${drag ? ' dragging' : ''}`} id="frame-inspector">
      <div className="rins-row">
        {/* The Scene and Outfit tiles are gone from here — they live in the
            console's `PlateRow` now, visible at rest. The card kept them for a
            version and they were two undiscovered gestures deep: the whole
            feature was filed as broken with the engine healthy underneath it.
            What stays is what only means something over the boxes: the weight
            every box is multiplied by, the arrangement, and the note that says
            which engine this run takes. */}
        <div className="opt n" id="g-region-base-wrap" data-lb="Global">
          <span className="lead">Global</span>
          <NumInput id="g-region-base" value={s.regionWeight} fine={0.05} bigStep={0.25}
                    title="Multiplies every region's LoRA strength at once. 1 uses the strengths as written."
                    onValue={s.setRegionWeight} />
        </div>

        <button className="opt ib" id="g-arrange" title="Distribute the boxes evenly"
                type="button" onClick={arrange.toggle}>
          <IconArrange />
        </button>
        {arrange.open && (
          <Menu anchor={arrange.anchor} onClose={arrange.close} items={[
            { label: 'Distribute in columns', run: () => s.setRegions(distribute(s.regions, true)) },
            { label: 'Distribute in rows', run: () => s.setRegions(distribute(s.regions, false)) },
          ]} />
        )}
      </div>

      {note && <p className="rins-note" id="region-note">{note}</p>}
    </div>
  )
}

/** Three decimals with the trailing zeros off, so a box at a half reads `0.5` and one
 *  mid-drag still reads to the precision the arrows step by. */
const fmt = (v: number) => v.toFixed(3).replace(/0+$/, '').replace(/\.$/, '')
