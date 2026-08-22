import { useCallback, useRef, useState } from 'react'

import { IconRegions } from '../icons'
import { dataUrl, shrinkB64 } from '../media/files'
import { loraIndex } from '../lora/tokens'
import { NEED_EDIT_LORA } from '../lora/note'
import { attached, newRegion, useStore, type EditMode, type Region } from '../store'
import { askResumeFocus } from './focus'
import { Inspector } from './Inspector'
import { MIN_SIDE, SNAP_EPS, SNAP_TO, clamp01, regionArmed, regionTag, snapEdge } from './geometry'

const HANDLES = ['nw', 'n', 'ne', 'e', 'se', 's', 'sw', 'w'] as const

/** How long a press has to be held to mean geometry where there is no ⌘ to hold.
 *  Apple's own long-press is 500ms and this is the same gesture, so it should not be a
 *  different length — a shorter one starts stealing taps. */
const LONG_PRESS_MS = 500

/**
 * The boxes, drawn on whatever they are drawn on — and, since the row and the map were
 * deleted, everything about them too.
 *
 * **This component is placed by its host rather than reparenting itself.** The vanilla
 * layer was one element moved between the frame and the first still with
 * `host.appendChild`, and losing it once cost three bugs that looked like three
 * features breaking: anything that replaced a container's innerHTML deleted it, and the
 * symptom was never "the layer is gone" — it was every caller of `drawRegions` dying at
 * its first statement, so the boxes vanished, the inspector would not close and the
 * Regional toggle appeared dead. Rendering it as a child of whichever host is showing
 * deletes the whole class of fault: React owns the placement, and `<Frame>` and the
 * first `.shot` each just include one.
 *
 * Every coordinate in here is a percentage, so nothing measures the host. The drag
 * measures this element's own rect, which is the host's content box — `inset: 0` makes
 * those the same box on the frame, and `bottom: var(--acts-h)` is what keeps them the
 * same box on a still, where `.shot` reserves padding under the picture for its two
 * buttons.
 *
 * Subscribed field by field rather than with a bare `useStore()`. It used to take the
 * whole store, so eight boxes and their card re-rendered on every keystroke in the main
 * prompt — survivable while this was a decoration over the canvas, and not while it is
 * the surface the feature lives on.
 */
export function RegionLayer({ over = 'frame' }: { over?: 'frame' | 'render' }) {
  // No flag anywhere: regions are on when a box exists. Both hosts mount this with
  // zero boxes too — the frame for the invite, the render because this layer is also
  // the surface ⌘-drag draws the *first* box on. What is left here is what to draw.
  const regions = useStore((z) => z.regions)
  const rsel = useStore((z) => z.rsel)
  const boxDrag = useStore((z) => z.boxDrag)
  const fileOver = useStore((z) => z.fileOver)
  const edit = useStore((z) => z.edit)
  const setEdit = useStore((z) => z.setEdit)
  const state = useStore((z) => z.state)
  const select = useStore((z) => z.select)
  const setRegions = useStore((z) => z.setRegions)
  const patchRegion = useStore((z) => z.patchRegion)
  const attach = useStore((z) => z.attach)

  const layer = useRef<HTMLDivElement>(null)
  const [guides, setGuides] = useState<{ v: number[]; h: number[] }>({ v: [], h: [] })
  const [dropHit, setDropHit] = useState<number | null>(null)
  /** Said on the layer rather than through `alert()`. A modal that stops the app to
   *  deliver one sentence is the wrong weight for it, and the sentence is about a
   *  control that is visible and dimmed an inch away. Cleared by the next thing you do,
   *  so there is no timer to get wrong and nothing to dismiss. */
  const [refused, setRefused] = useState<string | null>(null)
  const index = loraIndex(state)

  // What this layer *draws*, which is not whether it is listening — over a render it is
  // always listening, because a region you cannot touch is not addressable. See the
  // `edit` note in store.ts for the three states.
  //
  // The frame has nothing to protect: there is no render under it, it is the surface you
  // place boxes on, and the invitation lives there. So it is always geometry regardless
  // of the mode a previous render left behind.
  //
  // `fileOver` forces geometry anywhere. Dragging a photo over the window *is* the
  // question "where can this go", and without the boxes drawn the drop cannot be aimed
  // at one — which deletes the only moment anyone discovers that a box takes a
  // photograph at all. `boxDrag` holds geometry up through a gesture that is already
  // under way.
  const mode: EditMode =
    over === 'frame' || fileOver || boxDrag ? 'geometry' : edit

  const rectOf = () => layer.current?.getBoundingClientRect()
  const frameXY = (e: PointerEvent | React.PointerEvent | React.DragEvent): [number, number] => {
    const b = rectOf()
    if (!b) return [0, 0]
    return [clamp01((e.clientX - b.left) / b.width), clamp01((e.clientY - b.top) / b.height)]
  }

  const showGuides = (r: Region | null) => {
    if (!r) return setGuides({ v: [], h: [] })
    const near = (v: number) => SNAP_TO.some((c) => Math.abs(v - c) < 1e-6)
    setGuides({
      v: [r.x, r.x + r.w].filter(near),
      h: [r.y, r.y + r.h].filter(near),
    })
  }

  const onPointerDown = useCallback((e: React.PointerEvent<HTMLDivElement>) => {
    if (e.button !== 0) return
    const target = e.target as HTMLElement
    // The card and the frame button are children of this layer, so without this every
    // click on a numeric field or on the corner button would start drawing a rectangle
    // underneath it.
    if (target.closest('.rins,.rframe-btn')) return
    setRefused(null)
    const st = useStore.getState()
    // ⌘ means "a new one, here" and skips the hit test on purpose. Once a few
    // performers are placed there is often no bare canvas left to start a drag on, and
    // the alternative — move something out of the way, draw, move it back — is three
    // gestures to express one.
    const fresh = e.metaKey || e.ctrlKey
    const boxEl = fresh ? null : target.closest<HTMLElement>('.rbox')
    const handle = fresh ? null : (target.dataset.h ?? null)
    const [px, py] = frameXY(e)

    // Over a render, geometry is something you ask for. ⌘ asks for it — the same
    // modifier that already means "a new box, here" on the frame, so there is one rule
    // rather than two — and a long press asks for it where there is no modifier to
    // hold. Everything else is the frequent act: touch a performer, get their sentence.
    //
    // Nothing about this touches `regions`. The boxes are still there and still go with
    // the next run; what changes is only whether they are drawn.
    const live: EditMode =
      over === 'frame' || st.fileOver ? 'geometry' : st.edit
    if (live !== 'geometry') {
      if (fresh) {
        // Reveal, and nothing else. ⌘ *inside* geometry means "a new box, here" — so
        // letting this one press do both would answer "show me the boxes" by adding a
        // ninth one to the eight it just showed you. The gate and the gesture it gates
        // are two presses — *while there is something behind the gate*. With no boxes
        // at all, "show me the boxes" shows nothing, and a reveal-only press is a dead
        // press over exactly the surface ⌘-drag is supposed to draw on. So an empty
        // set falls through and this one gesture draws the first box.
        st.setEdit('geometry')
        if (st.regions.length) return
      } else {
        // Resolved on release, not on press, so one gesture can still become the other:
        // a quick tap is "open this", the same press held is "let me move things". On a
        // mouse the release is imperceptible, so the card still feels like it opens on
        // the click.
        const el = layer.current
        const i = boxEl ? Number(boxEl.dataset.i) : -1
        let held = false
        const hold = window.setTimeout(() => {
          held = true
          useStore.getState().setEdit('geometry')
        }, LONG_PRESS_MS)
        const cancel = () => {
          window.clearTimeout(hold)
          el?.removeEventListener('pointerup', done)
          el?.removeEventListener('pointercancel', cancel)
          el?.removeEventListener('pointermove', moved)
        }
        // A press that travels is a scroll or a slip, never a tap.
        const moved = (ev: PointerEvent) => {
          if (Math.abs(ev.clientX - e.clientX) + Math.abs(ev.clientY - e.clientY) > 8) cancel()
        }
        const done = () => {
          cancel()
          if (held || i < 0) return
          // The card opens with its caret in the sentence, because opening it is only
          // half the instruction and the other half is that you can start typing —
          // which is the whole edit-and-regenerate loop.
          askResumeFocus()
          const now = useStore.getState()
          now.select(i)
          now.setEdit('content')
        }
        el?.addEventListener('pointerup', done)
        el?.addEventListener('pointercancel', cancel)
        el?.addEventListener('pointermove', moved)
        return
      }
    }

    let idx: number
    let grip: string
    let orig: Region
    let grab: [number, number] = [0, 0]

    if (boxEl) {
      idx = Number(boxEl.dataset.i)
      const r = st.regions[idx]
      if (!r) return
      grip = handle ?? 'move'
      orig = { ...r }
      grab = [px - orig.x, py - orig.y]
      st.select(idx)
    } else {
      // A drag on bare canvas draws a new box. Capped, and silently — the cap is the
      // backend's and there is nothing useful to say about it mid-gesture.
      if (st.regions.length >= (st.state?.max_regions ?? 8)) return
      const r = newRegion({ x: px, y: py, w: 0, h: 0 })
      idx = st.regions.length
      grip = 'se'
      orig = { ...r }
      useStore.setState({ regions: [...st.regions, r], rsel: idx })
    }

    e.preventDefault()
    // Capture so a drag that leaves the frame still tracks and still ends: with plain
    // listeners a release outside the window leaves the box stuck to the cursor.
    // Guarded because a pointer already gone by the time we ask throws NotFoundError,
    // and losing the capture is survivable where losing the rest of this handler is
    // not.
    const el = layer.current
    try { el?.setPointerCapture(e.pointerId) } catch { /* gone already */ }
    useStore.getState().setBoxDrag(true)

    // One store write per animation frame, however many pointermove events land inside
    // it. The card's four numbers have to move while you drag — that is the whole
    // reason they survived the row they used to live in — so this cannot defer to
    // pointerup; what it can do is stop asking React to paint twice for one frame of
    // pointer input.
    let pending: Region | null = null
    let raf = 0
    const flush = () => {
      raf = 0
      if (!pending) return
      useStore.getState().patchRegion(idx, pending)
      showGuides(pending)
      pending = null
    }

    const move = (ev: PointerEvent) => {
      const [x, y] = frameXY(ev)
      const alt = ev.altKey
      const live = useStore.getState().regions
      const next: Region = { ...orig }
      if (grip === 'move') {
        next.x = Math.min(Math.max(snapEdge(x - grab[0], 'x', live, idx, alt), 0), 1 - orig.w)
        next.y = Math.min(Math.max(snapEdge(y - grab[1], 'y', live, idx, alt), 0), 1 - orig.h)
        next.w = orig.w
        next.h = orig.h
      } else {
        let l = orig.x
        let t = orig.y
        let rt = orig.x + orig.w
        let bt = orig.y + orig.h
        if (grip.includes('w')) l = snapEdge(x, 'x', live, idx, alt)
        if (grip.includes('e')) rt = snapEdge(x, 'x', live, idx, alt)
        if (grip.includes('n')) t = snapEdge(y, 'y', live, idx, alt)
        if (grip.includes('s')) bt = snapEdge(y, 'y', live, idx, alt)
        // Sorted rather than clamped, so dragging a handle past its opposite flips the
        // box the way every other editor does instead of jamming.
        next.x = Math.min(l, rt)
        next.w = Math.abs(rt - l)
        next.y = Math.min(t, bt)
        next.h = Math.abs(bt - t)
      }
      pending = next
      if (!raf) raf = requestAnimationFrame(flush)
    }

    const up = (ev: PointerEvent) => {
      el?.removeEventListener('pointermove', move)
      el?.removeEventListener('pointerup', up)
      el?.removeEventListener('pointercancel', up)
      try { el?.releasePointerCapture(ev.pointerId) } catch { /* gone already */ }
      if (raf) cancelAnimationFrame(raf)
      flush()
      showGuides(null)
      const r = useStore.getState().regions[idx]
      if (r) {
        // A click rather than a drag on bare canvas leaves a zero-area box, which the
        // backend rejects outright. Grow it to something usable instead of erroring at
        // Generate about a rectangle nobody meant to make.
        if (r.w < MIN_SIDE || r.h < MIN_SIDE) {
          const grown = grip === 'se' && !orig.w
            ? { w: Math.max(r.w, 0.28), h: Math.max(r.h, 0.6) }
            : { w: Math.max(r.w, MIN_SIDE), h: Math.max(r.h, MIN_SIDE) }
          useStore.getState().patchRegion(idx, {
            ...grown,
            x: Math.min(r.x, 1 - grown.w),
            y: Math.min(r.y, 1 - grown.h),
          })
        }
        useStore.getState().select(idx)
        // Selecting is not focusing, and this is the difference between the keyboard
        // working and not. The `preventDefault` above is what owns the drag, and it
        // also suppresses the focus the click would have given the box — so ⌫ and the
        // arrows, which are handled on this layer and find their target with
        // `closest('.rbox')`, only ever reached a box you had *tabbed* to. Clicking one
        // and pressing delete did nothing at all. It matters more now than it did: Tab
        // between boxes is what replaced the map's `‹ 2/5 ›`.
        //
        // Which of the two gets it depends on what just happened, and this handler is
        // the only place that knows: a box you have just *drawn* wants the caret in its
        // sentence, because the card appearing is only half the instruction and the
        // other half is that you can start typing. One you merely moved wants the
        // rectangle, so the next key still reaches it. Deferred a frame for the first,
        // because the field does not exist until React has painted the card.
        if (grip === 'se' && !orig.w) {
          requestAnimationFrame(() =>
            document.querySelector<HTMLInputElement>('#r-prompt')?.focus())
        } else {
          el?.querySelector<HTMLElement>(`.rbox[data-i="${idx}"]`)?.focus()
        }
      }
      useStore.getState().setBoxDrag(false)
    }

    el?.addEventListener('pointermove', move)
    el?.addEventListener('pointerup', up)
    el?.addEventListener('pointercancel', up)
  }, [])

  /* Dropping onto a box is the gesture the box exists for; dropping onto bare canvas is
     the scene, which is a different thing entirely and gated on a weight that may not
     be downloaded. One handler, because the target decides. */
  const onDrop = async (e: React.DragEvent) => {
    e.preventDefault()
    setDropHit(null)
    const f = e.dataTransfer.files[0]
    if (!f?.type.startsWith('image/')) return
    const hit = (e.target as HTMLElement).closest<HTMLElement>('.rbox')
    // Said, not swallowed. Without the edit LoRA this used to return in silence, which
    // is indistinguishable from a drop the page never received — and the target is
    // visibly lit, so refusing quietly is a promise made and broken.
    if (!hit && !state?.edit_lora) {
      setRefused(NEED_EDIT_LORA)
      return
    }
    setRefused(null)
    const b64 = await shrinkB64(f)
    if (!b64) return
    if (hit) {
      const i = Number(hit.dataset.i)
      attach(i, 'identity', b64)
      select(i)
    } else {
      attach('frame', 'scene', b64)
      // And show the frame's card, because this drop just moved the run onto the
      // krea2edit compose — a different engine that regenerates the whole frame and
      // is several times slower. The plate itself becomes visible behind the boxes,
      // but no arrangement of rectangles shows what it costs, and the sentence that
      // does lives in that card. Attaching a plate from inside the card is already
      // self-explaining; this is the path that was not.
      select(-1)
    }
  }

  // The even split, from the cold canvas. Two rectangles appearing is the whole
  // instruction — the same seed the console button used to plant, now reachable from
  // the surface the boxes live on. Focus lands in the first box's sentence, because
  // the card opening is only half the instruction and the other half is that you can
  // start typing.


  return (
    <div id="region-layer" ref={layer}
         /* Bare frame under a LoRA drag says what letting go there would do. It is
            the layer rather than the frame that carries it because the frame does
            not know a drag is happening — and it is suppressed the moment a box is
            under the cursor, so the two captions are never on screen together. */
         className={mode}
         onPointerDown={onPointerDown}
         onDragOver={(e) => {
           e.preventDefault()
           // Re-read on every `dragover` rather than latched on enter: a drag can
           // begin outside the layer, and this is the event that repeats.
           const hit = (e.target as HTMLElement).closest<HTMLElement>('.rbox')
           // Only the box under the cursor names itself. Eight captions on eight boxes
           // is the same wall of text the per-region rows were removed for, drawn on
           // the picture this time.
           setDropHit(hit ? Number(hit.dataset.i) : null)
         }}
         onDragLeave={(e) => {
           if (!e.currentTarget.contains(e.relatedTarget as Node)) setDropHit(null)
         }}
         onDrop={(e) => { void onDrop(e) }}
         onKeyDown={(e) => {
           // Escape puts the picture back. Over a render that means all the way out —
           // one press returns the clean result, which is the state the canvas is meant
           // to be in — and on the frame, where there is no render to return to, it
           // means the frame's own card, the keyboard half of the button in the corner.
           // Handled before the `.rbox` test so it works from inside the card's fields.
           if (e.key === 'Escape') {
             e.preventDefault()
             select(-1)
             if (over === 'render') setEdit('off')
             return
           }
           const el = (e.target as HTMLElement).closest<HTMLElement>('.rbox')
           if (!el) return
           const i = Number(el.dataset.i)
           const r = regions[i]
           if (!r) return
           if (e.key === 'Backspace' || e.key === 'Delete') {
             e.preventDefault()
             setRegions(regions.filter((_, n) => n !== i))
             select(Math.min(i, regions.length - 2))
             return
           }
           const d = { ArrowLeft: [-1, 0], ArrowRight: [1, 0], ArrowUp: [0, -1], ArrowDown: [0, 1] }[e.key]
           if (!d) return
           e.preventDefault()
           // Steps match the card's own fine and coarse steps.
           const k = e.metaKey || e.ctrlKey ? 0.1 : 0.01
           patchRegion(i, {
             x: Math.min(Math.max(r.x + (d[0] ?? 0) * k, 0), 1 - r.w),
             y: Math.min(Math.max(r.y + (d[1] ?? 0) * k, 0), 1 - r.h),
           })
         }}
         onFocus={(e) => {
           const el = (e.target as HTMLElement).closest<HTMLElement>('.rbox')
           if (el && Number(el.dataset.i) !== rsel) select(Number(el.dataset.i))
         }}>
      {regions.map((r, i) => {
        const tag = regionTag(index, r)
        const face = attached(r, 'identity')
        return (
          <div key={r.id} className={['rbox', regionArmed(index, r) ? 'armed' : '',
                                      i === rsel || i === dropHit ? 'sel' : '',
                                      i === dropHit ? 'drop-hit' : ''].filter(Boolean).join(' ')}
               data-i={i} tabIndex={0}
               /* What dropping this would do. One caption now rather than two: a
                  photograph is always the likeness, and a LoRA is no longer
                  something you can drop — it is picked from the card's own
                  dropdown, which says which box by being on it. */
               data-drop="This character"
               style={{
                 left: `${clamp01(r.x) * 100}%`,
                 top: `${clamp01(r.y) * 100}%`,
                 width: `${Math.min(1 - clamp01(r.x), clamp01(r.w)) * 100}%`,
                 height: `${Math.min(1 - clamp01(r.y), clamp01(r.h)) * 100}%`,
               }}>
            {face && <img className="face" src={dataUrl(face)} alt="" />}
            {tag.text && (
              <span className="tag">{tag.muted ? <em>{tag.text}</em> : tag.text}</span>
            )}
            {HANDLES.map((hd) => <i key={hd} data-h={hd} />)}
          </div>
        )
      })}

      {/* Drawn only while a drag is landing on one, so the line is feedback rather than
          furniture. */}
      {guides.v.map((v) => <div key={`v${v}`} className="guide v" style={{ left: `${v * 100}%` }} />)}
      {guides.h.map((v) => <div key={`h${v}`} className="guide h" style={{ top: `${v * 100}%` }} />)}

      <Inspector mode={mode} />

      {/* The frame's own affordance, because a tap on bare canvas is already taken — it
          draws a rectangle. Scene, outfit and the region weight belong to every box at
          once, so they need somewhere that is not any one of them, and the frame is
          that somewhere. The glyph is the one you pressed to arm the mode, which is
          what makes it legible here: an icon can carry a control whose glyph is
          unambiguous and whose home you are already in. */}
      {!!regions.length && mode === 'geometry' && (
        <button className={`rframe-btn${rsel < 0 ? ' on' : ''}`} type="button" id="g-frame"
                title="The frame — scene, outfit, and how hard every box pushes"
                onClick={() => select(rsel < 0 ? 0 : -1)}>
          <IconRegions />
        </button>
      )}

      {refused && <p className="rins-refusal">{refused}</p>}
    </div>
  )
}

/** Exported for the snap-guide threshold, which the card's coarse step wants to stay
 *  clear of. */
export { SNAP_EPS }
