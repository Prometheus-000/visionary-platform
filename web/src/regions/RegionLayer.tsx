import { useCallback, useRef, useState } from 'react'

import { dataUrl, shrinkB64 } from '../media/files'
import { loraIndex } from '../lora/tokens'
import { NEED_EDIT_LORA } from '../lora/note'
import { newRegion, useStore, type Region } from '../store'
import { MIN_SIDE, SNAP_EPS, SNAP_TO, clamp01, regionArmed, regionTag, snapEdge } from './geometry'

const HANDLES = ['nw', 'n', 'ne', 'e', 'se', 's', 'sw', 'w'] as const

/**
 * The boxes, drawn on whatever they are drawn on.
 *
 * **This component is placed by its host rather than reparenting itself.** The
 * vanilla layer was one element moved between the frame and the first still with
 * `host.appendChild`, and losing it once cost three bugs that looked like three
 * features breaking: anything that replaced a container's innerHTML deleted it, and
 * the symptom was never "the layer is gone" — it was every caller of `drawRegions`
 * dying at its first statement, so the boxes vanished, the inspector would not
 * close and the Regional toggle appeared dead. Rendering it as a child of whichever
 * host is showing deletes the whole class of fault: React owns the placement, and
 * `<Frame>` and the first `.shot` each just include one.
 *
 * Every coordinate in here is a percentage, so nothing measures the host. The drag
 * measures this element's own rect, which is the host's content box — `inset: 0`
 * makes those the same box.
 */
export function RegionLayer() {
  const s = useStore()
  const layer = useRef<HTMLDivElement>(null)
  const [guides, setGuides] = useState<{ v: number[]; h: number[] }>({ v: [], h: [] })
  const [dropHit, setDropHit] = useState<number | null>(null)
  const index = loraIndex(s.state)

  // Four reasons the boxes are on screen, and they are all "you are working on a
  // region right now": the caret is in the region bar, a box is mid-drag, a file is
  // over the window, or regions were just armed and the two seeded rectangles are
  // the instruction. Anything else — including looking at the render you just made
  // — and they go.
  //
  // Armed used to mean visible, which meant a continuous white rectangle across
  // every picture you rendered in the mode you render most in: chrome painted over
  // the one thing the page exists to show. Gating on focus in the region bar was
  // the overcorrection — the boxes *are* the list, and a list you have to click a
  // text field to see is a list you cannot use. So a finished render is the only
  // thing that puts them away, and the very next sign you are making the next one
  // brings them back.
  const visible = s.regional && (s.regionPeek || !s.freshRender)

  const rectOf = () => layer.current?.getBoundingClientRect()
  const frameXY = (e: PointerEvent | React.PointerEvent): [number, number] => {
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
    const st = useStore.getState()
    // ⌘ means "a new one, here" and skips the hit test on purpose. Once a few
    // performers are placed there is often no bare canvas left to start a drag on,
    // and the alternative — move something out of the way, draw, move it back — is
    // three gestures to express one.
    const fresh = e.metaKey || e.ctrlKey
    const target = e.target as HTMLElement
    const boxEl = fresh ? null : target.closest<HTMLElement>('.rbox')
    const handle = fresh ? null : (target.dataset.h ?? null)
    const [px, py] = frameXY(e)

    let idx: number
    let mode: string
    let orig: Region
    let grab: [number, number] = [0, 0]

    if (boxEl) {
      idx = Number(boxEl.dataset.i)
      const r = st.regions[idx]
      if (!r) return
      mode = handle ?? 'move'
      orig = { ...r }
      grab = [px - orig.x, py - orig.y]
      st.select(idx)
    } else {
      // A drag on bare canvas draws a new box. Capped, and silently — the cap is
      // the backend's and there is nothing useful to say about it mid-gesture.
      if (st.regions.length >= (st.state?.max_regions ?? 8)) return
      const r = newRegion({ x: px, y: py, w: 0, h: 0 })
      idx = st.regions.length
      mode = 'se'
      orig = { ...r }
      useStore.setState({ regions: [...st.regions, r], rsel: idx })
    }

    e.preventDefault()
    // Capture so a drag that leaves the frame still tracks and still ends: with
    // plain listeners a release outside the window leaves the box stuck to the
    // cursor. Guarded because a pointer already gone by the time we ask throws
    // NotFoundError, and losing the capture is survivable where losing the rest of
    // this handler is not.
    const el = layer.current
    try { el?.setPointerCapture(e.pointerId) } catch { /* gone already */ }
    useStore.getState().setRegionPeek(true)

    // One store write per animation frame, however many pointermove events land
    // inside it. The inspector's numbers have to move while you drag — that is why
    // they survived the per-region rows — so this cannot defer to pointerup; what
    // it can do is stop asking React to paint twice for one frame of pointer input.
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
      const regions = useStore.getState().regions
      const next: Region = { ...orig }
      if (mode === 'move') {
        next.x = Math.min(Math.max(snapEdge(x - grab[0], 'x', regions, idx, alt), 0), 1 - orig.w)
        next.y = Math.min(Math.max(snapEdge(y - grab[1], 'y', regions, idx, alt), 0), 1 - orig.h)
        next.w = orig.w
        next.h = orig.h
      } else {
        let l = orig.x
        let t = orig.y
        let rt = orig.x + orig.w
        let bt = orig.y + orig.h
        if (mode.includes('w')) l = snapEdge(x, 'x', regions, idx, alt)
        if (mode.includes('e')) rt = snapEdge(x, 'x', regions, idx, alt)
        if (mode.includes('n')) t = snapEdge(y, 'y', regions, idx, alt)
        if (mode.includes('s')) bt = snapEdge(y, 'y', regions, idx, alt)
        // Sorted rather than clamped, so dragging a handle past its opposite flips
        // the box the way every other editor does instead of jamming.
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
        // A click rather than a drag on bare canvas leaves a zero-area box, which
        // the backend rejects outright. Grow it to something usable instead of
        // erroring at Generate about a rectangle nobody meant to make.
        if (r.w < MIN_SIDE || r.h < MIN_SIDE) {
          const grown = mode === 'se' && !orig.w
            ? { w: Math.max(r.w, 0.28), h: Math.max(r.h, 0.6) }
            : { w: Math.max(r.w, MIN_SIDE), h: Math.max(r.h, MIN_SIDE) }
          useStore.getState().patchRegion(idx, {
            ...grown,
            x: Math.min(r.x, 1 - grown.w),
            y: Math.min(r.y, 1 - grown.h),
          })
        }
        useStore.getState().select(idx)
      }
      // The hold only had to survive the pointer being down; from here the boxes
      // stay up on their own because the selection puts the caret in the bar.
      useStore.getState().setRegionPeek(false)
    }

    el?.addEventListener('pointermove', move)
    el?.addEventListener('pointerup', up)
    el?.addEventListener('pointercancel', up)
  }, [])

  /* Dropping onto a box is the gesture the box exists for; dropping onto bare
     canvas is the scene, which is a different thing entirely and gated on a weight
     that may not be downloaded. One handler, because the target decides. */
  const onDrop = async (e: React.DragEvent) => {
    e.preventDefault()
    setDropHit(null)
    const f = e.dataTransfer.files[0]
    if (!f?.type.startsWith('image/')) return
    const hit = (e.target as HTMLElement).closest<HTMLElement>('.rbox')
    // Said, not swallowed. Without the edit LoRA this used to return in silence,
    // which is indistinguishable from a drop the page never received — and the
    // target is visibly lit, so refusing quietly is a promise made and broken.
    if (!hit && !s.state?.edit_lora) {
      alert(NEED_EDIT_LORA)
      return
    }
    const b64 = await shrinkB64(f)
    if (!b64) return
    if (hit) {
      const i = Number(hit.dataset.i)
      s.patchRegion(i, { ref: b64 })
      s.select(i)
    } else {
      s.setPlate('scene', b64)
    }
  }

  return (
    <div id="region-layer" ref={layer}
         className={[visible ? 'show' : '', s.regional ? '' : 'off'].filter(Boolean).join(' ')}
         onPointerDown={onPointerDown}
         onDragOver={(e) => {
           if (!s.regional) return
           e.preventDefault()
           const hit = (e.target as HTMLElement).closest<HTMLElement>('.rbox')
           // Only the box under the cursor names itself. Eight captions on eight
           // boxes is the same wall of text the per-region rows were removed for,
           // drawn on the picture this time.
           setDropHit(hit ? Number(hit.dataset.i) : null)
         }}
         onDragLeave={(e) => {
           if (!e.currentTarget.contains(e.relatedTarget as Node)) setDropHit(null)
         }}
         onDrop={(e) => { void onDrop(e) }}
         onKeyDown={(e) => {
           const el = (e.target as HTMLElement).closest<HTMLElement>('.rbox')
           if (!el) return
           const i = Number(el.dataset.i)
           const r = s.regions[i]
           if (!r) return
           if (e.key === 'Backspace' || e.key === 'Delete') {
             e.preventDefault()
             s.setRegions(s.regions.filter((_, n) => n !== i))
             s.select(Math.min(i, s.regions.length - 2))
             return
           }
           const d = { ArrowLeft: [-1, 0], ArrowRight: [1, 0], ArrowUp: [0, -1], ArrowDown: [0, 1] }[e.key]
           if (!d) return
           e.preventDefault()
           // Steps match the inspector cells' own fine and coarse steps.
           const k = e.metaKey || e.ctrlKey ? 0.1 : 0.01
           s.patchRegion(i, {
             x: Math.min(Math.max(r.x + (d[0] ?? 0) * k, 0), 1 - r.w),
             y: Math.min(Math.max(r.y + (d[1] ?? 0) * k, 0), 1 - r.h),
           })
         }}
         onFocus={(e) => {
           const el = (e.target as HTMLElement).closest<HTMLElement>('.rbox')
           if (el && Number(el.dataset.i) !== s.rsel) s.select(Number(el.dataset.i))
         }}>
      {s.regions.map((r, i) => {
        const tag = regionTag(index, r)
        return (
          <div key={r.id} className={['rbox', regionArmed(index, r) ? 'armed' : '',
                                      i === s.rsel || i === dropHit ? 'sel' : '',
                                      i === dropHit ? 'drop-hit' : ''].filter(Boolean).join(' ')}
               data-i={i} tabIndex={0} data-drop="This character"
               style={{
                 left: `${clamp01(r.x) * 100}%`,
                 top: `${clamp01(r.y) * 100}%`,
                 width: `${Math.min(1 - clamp01(r.x), clamp01(r.w)) * 100}%`,
                 height: `${Math.min(1 - clamp01(r.y), clamp01(r.h)) * 100}%`,
               }}>
            {r.ref && <img className="face" src={dataUrl(r.ref)} alt="" />}
            {tag.text && (
              <span className="tag">{tag.muted ? <em>{tag.text}</em> : tag.text}</span>
            )}
            {HANDLES.map((hd) => <i key={hd} data-h={hd} />)}
          </div>
        )
      })}
      {/* Drawn only while a drag is landing on one, so the line is feedback rather
          than furniture. */}
      {guides.v.map((v) => <div key={`v${v}`} className="guide v" style={{ left: `${v * 100}%` }} />)}
      {guides.h.map((v) => <div key={`h${v}`} className="guide h" style={{ top: `${v * 100}%` }} />)}
    </div>
  )
}

/** Exported for the snap-guide threshold, which the inspector's coarse step wants
 *  to stay clear of. */
export { SNAP_EPS }
