import { useEffect, useRef } from 'react'

import { IconArrange, IconOutfit, IconPhoto, IconScene, IconTrash } from '../icons'
import { Menu } from '../ui/Menu'
import { NumInput } from '../ui/NumInput'
import { usePopover } from '../ui/Popover'
import { DropTile } from '../media/DropTile'
import { shrinkB64 } from '../media/files'
import { caretProps, dropCaret } from '../lora/caret'
import { NEED_EDIT_LORA, regionNote } from '../lora/note'
import { chipFrom, loraIndex } from '../lora/tokens'
import { attached, regionsLive, useStore, type EditMode, type Region } from '../store'
import { takeResumeFocus } from './focus'
import { clamp01, distribute, readRegions } from './geometry'

const PLATE_TITLE = {
  scene: 'Scene photo. The picture is generated inside it — lighting, perspective '
    + 'and shadows integrate.',
  outfit: 'Outfit or object photo. Transferred onto the subjects rather than pasted '
    + 'into the frame.',
} as const

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
  if (!r) return mode === 'geometry' ? <FrameCard drag={s.boxDrag} /> : null
  return <BoxCard r={r} i={s.rsel} drag={s.boxDrag} mode={mode} />
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
            LoRA is the dropdown below, which says *which box* by being on it. */}
        <div className="opt wide">
          <input id="r-prompt" ref={field} value={r.prompt}
                 placeholder="a man in a denim jacket, laughing"
                 title="This performer only — who they are and what they are doing. The scene, the light and the lens go in the prompt below; where they stand is the box. Do not write a position here, it is already said."
                 onChange={(e) => s.patchRegion(i, { prompt: e.target.value })}
                 {...caretProps('region', (v) => useStore.getState().patchRegion(
                   useStore.getState().rsel, { prompt: v }))} />
        </div>
      </div>

      <div className="rins-row">
        <DropTile id="r-ref" label="Photo" value={attached(r, 'identity')} glyph={<IconPhoto />}
                  title="A photo of this character. Pulls the box toward that likeness during sampling — stacks with the LoRA, and works without one."
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

      {/* The escape hatch, and during a drag the readout that teaches what it is an
          escape from. Dragging is what makes "0.5 0 0.5 1" mean a rectangle; the
          numbers never taught the dragging, which is why they are the second row here
          and were the wrong primary in the row this replaced.

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
    </div>
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
        {/* The plates ride on two conditions, not one: regions on, and the weight they
            need actually downloaded. The two get different treatments, and the split
            is the point. Regions off: this card does not exist, because there is
            nothing to put a scene behind. Weight missing: the tiles are *locked and
            dimmed*, because an install without the edit LoRA is one download away from
            scene and outfit transfer and had no way of learning either existed. A
            model-gated control the model will never read stays hidden; a weight-gated
            control is not that — it is a purchase you have not made yet, and hiding it
            hides the decision rather than the capability.

            Region photos ride on neither: a mold is not an extra_ref plate, so it
            needs no edit LoRA and never switches paths. */}
        {(['scene', 'outfit'] as const).map((slot) => (
          <DropTile key={slot} id={`g-drop-${slot}`} label={slot === 'scene' ? 'Scene' : 'Outfit'}
                    value={attached(s.frame, slot)} locked={!s.state?.edit_lora}
                    glyph={slot === 'scene' ? <IconScene /> : <IconOutfit />}
                    title={s.state?.edit_lora ? PLATE_TITLE[slot] : NEED_EDIT_LORA}
                    onFile={async (f) => {
                      const b64 = await shrinkB64(f)
                      if (b64) s.attach('frame', slot, b64)
                    }}
                    onClear={() => s.attach('frame', slot, null)} />
        ))}

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
