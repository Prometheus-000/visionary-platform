import { useCallback, useEffect, useRef, useState } from 'react'

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

/** How far a press may travel and still be a click. Shared by the two questions that
 *  ask it — *was that a tap or a scroll* over a render, and *was that a click or a
 *  drag* while framing — because two thresholds for one gesture is two gestures. */
const CLICK_SLOP = 8

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

  /**
   * Did this really happen *on the layer*?
   *
   * React events bubble along the **React tree**, and `Popover` is a portal to
   * `<body>` — so the LoRA menu is a DOM child of the document and a React child
   * of the card, which is a React child of this div. Every handler below was
   * written against the DOM, and the one that mattered guarded with
   * `closest('.rins')`: true of the card, false of a menu opened *from* the card,
   * because the menu is not inside it.
   *
   * So pressing a row in the LoRA picker arrived here as a press on the picture.
   * It drew a rectangle, `preventDefault()`ed the pointerdown — which is what
   * suppresses the compatibility mousedown/mouseup — and captured the pointer, so
   * the click retargeted to this div and the row's own handler never ran. The
   * symptom was the picker doing nothing except quietly adding a region.
   *
   * The DOM is the authority on what is on the layer, so ask it. A `closest()`
   * denylist cannot answer this: it would have to name every portal anyone ever
   * opens from inside a region.
   */
  const onLayer = useCallback(
    (e: React.SyntheticEvent) => !!layer.current?.contains(e.target as Node), [])
  const [guides, setGuides] = useState<{ v: number[]; h: number[] }>({ v: [], h: [] })
  const [dropHit, setDropHit] = useState<number | null>(null)
  /** Said on the layer rather than through `alert()`. A modal that stops the app to
   *  deliver one sentence is the wrong weight for it, and the sentence is about a
   *  control that is visible and dimmed an inch away. Cleared by the next thing you do,
   *  so there is no timer to get wrong and nothing to dismiss. */
  const [refused, setRefused] = useState<string | null>(null)
  /** Which box the pointer is over, decided by `hitAt` rather than by `:hover`. It has
   *  to be ours: CSS hover follows paint order, so the hairline that says "this is what
   *  you would be touching" would name a different box than the click opens — and that
   *  hairline is the only thing standing in for boxes that are deliberately not drawn.
   *  The ref is what `hitAt` reads, because it runs inside handlers that were closed
   *  over on mount. */
  const [under, setUnder] = useState(-1)
  const underRef = useRef(-1)
  const setUnderTo = (i: number) => {
    if (underRef.current === i) return
    underRef.current = i
    setUnder(i)
  }
  const index = loraIndex(state)

  // **A press that is not on the card closes the card.** The rule is stated as one
  // exception rather than as a list of places, because "anywhere outside the card"
  // does not stop at the edge of the picture — the console, the strip and the page
  // are outside it too, and a card that survived a press on the prompt field would be
  // a panel you have to come back and dismiss.
  //
  // `.menu` is the second exception and it is not a special case, it is the same one:
  // `Popover` portals to `<body>`, so the LoRA picker opened *from* the card is a DOM
  // sibling of it. Closing the card under the row you are pressing takes the row with
  // it, and the press lands on whatever the menu was covering. The layer's own
  // handlers already learned this once — see `onLayer`.
  //
  // Capture, so it has run before the layer's own `onPointerDown` decides what the
  // press means. A click that opens a card therefore closes it first, which is what
  // makes clicking from one box to another swap cards rather than toggle one off.
  useEffect(() => {
    const shut = (e: PointerEvent) => {
      if ((e.target as HTMLElement | null)?.closest('.rins,.menu')) return
      useStore.getState().setCardOpen(false)
    }
    document.addEventListener('pointerdown', shut, true)
    return () => document.removeEventListener('pointerdown', shut, true)
  }, [])

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
  // `boxDrag` holds geometry up through a gesture already under way — a render that
  // lands mid-drag sets `off` underneath it — but it must not *promote* one: a drag
  // that began in `content` is the open box being adjusted, and flipping the layer to
  // geometry for its duration would flash every other box onto the render.
  const mode: EditMode =
    over === 'frame' || fileOver || (boxDrag && edit !== 'content') ? 'geometry' : edit

  const rectOf = () => layer.current?.getBoundingClientRect()
  const frameXY = (
    e: PointerEvent | React.PointerEvent | React.MouseEvent | React.DragEvent,
  ): [number, number] => {
    const b = rectOf()
    if (!b) return [0, 0]
    return [clamp01((e.clientX - b.left) / b.width), clamp01((e.clientY - b.top) / b.height)]
  }

  /** Every part of this layer under a point, topmost first. The stack rather than the
   *  target, because the target is whatever painted last and that is the one fact the
   *  hit test must not consult. See `hitAt`. */
  const stackAt = (cx: number, cy: number) => {
    const out: HTMLElement[] = []
    const el = layer.current
    if (!el) return out
    for (const n of document.elementsFromPoint(cx, cy)) {
      if (n === el) break
      out.push(n as HTMLElement)
    }
    return out
  }

  /**
   * The smallest box under the point, or -1 for bare canvas.
   *
   * **The canvas has no layers, so the boxes do not have them either.** They are areas
   * of one picture, not a stack of cards, and nothing in this feature ever says which
   * of two overlapping rectangles is on top — but they are absolutely-positioned
   * siblings, so the DOM had an opinion anyway and it was the worst one available:
   * whichever was drawn *last* took every click, every hover and every drop. A
   * performer placed inside a wide background box could be reached and the background
   * box could not, and the eight handles of anything underneath were simply gone.
   * Nobody chose that ordering; it fell out of `regions` being an array, which is a
   * storage detail the picture should never have been able to see.
   *
   * What replaces it is the rule Phase 6 already states for exactly this ambiguity —
   * *resolve toward the smaller object* — because widening a selection is cheap and
   * obvious while guessing large silently edits the wrong scope. Two boxes of the same
   * size still need a tiebreak and the array order is as good as any; what matters is
   * that it is no longer the *first* question.
   */
  const boxAt = (stack: HTMLElement[]) => {
    const live = useStore.getState().regions
    let best = -1
    let area = Infinity
    for (const n of stack) {
      if (!n.classList.contains('rbox')) continue
      const r = live[Number(n.dataset.i)]
      if (!r || r.w * r.h >= area) continue
      area = r.w * r.h
      best = Number(n.dataset.i)
    }
    return best
  }

  /**
   * The handle under the point, whichever box owns it.
   *
   * `boxAt`'s rule, and it has to be the same one: a handle is smaller than any box,
   * so it resolves first. What this replaces asked a second question after that —
   * *is this handle already lit* — and the answer put the layers straight back in.
   * Lighting is `sel` or `under`, `under` is decided by the **bodies**, and a box
   * lying beneath a smaller one is never what a body-hover names. So its eight dots
   * never lit, and a press landing exactly on one of them fell through to the box on
   * top and *moved* it. A wide background box with a performer sitting on its right
   * edge could not be widened at all: the press that should have taken its east
   * handle slid the performer sideways instead.
   *
   * The objection that gate was there for — *an invisible handle is not an object,
   * and letting one win gives every box four edges of theft at a target nobody can
   * see* — is answered by disclosing rather than by refusing. Hover resolves through
   * this same function, so crossing a handle lights that box's dots before any press,
   * and the pointer sits on the thing it is about to grab. Outside geometry the
   * handles are `display:none`, which keeps them out of `elementsFromPoint` entirely,
   * so this can only ever name a box that is drawn.
   */
  const handleAt = (stack: HTMLElement[]) => {
    const live = useStore.getState().regions
    let best: { i: number; h: string } | null = null
    let area = Infinity
    for (const n of stack) {
      if (!n.dataset.h) continue
      const i = Number(n.closest<HTMLElement>('.rbox')?.dataset.i)
      const r = live[i]
      if (!r || r.w * r.h >= area) continue
      area = r.w * r.h
      best = { i, h: n.dataset.h }
    }
    return best
  }

  /** Which box a point addresses, and which of its handles if any. One `stackAt` for
   *  both questions: this runs on every pointermove. */
  const hitAt = (cx: number, cy: number): { i: number; h: string | null } => {
    const stack = stackAt(cx, cy)
    return handleAt(stack) ?? { i: boxAt(stack), h: null }
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
    if (!onLayer(e)) return
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
    // Not `target.closest('.rbox')`. The target is the topmost thing under the cursor,
    // which is the array order wearing a costume — see `boxAt`.
    const hit = fresh ? { i: -1, h: null } : hitAt(e.clientX, e.clientY)
    const handle = hit.h
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
    // The one exception, and it is the whole of the second change: **an open box is
    // adjustable.** Editing what is inside a box and redrawing the box are still two
    // different acts, and the reason the second one was gated was that arming it used
    // to put eight rectangles and their handles over a render you were judging. That
    // objection is about boxes you have *not* touched — and it is already answered,
    // because nothing is drawn until you click. Having clicked one, being told to press
    // again with a modifier held to move the rectangle you are looking at is friction
    // with nothing behind it. So the box whose card is open behaves exactly as it does
    // in geometry; every other box stays undrawn and stays a tap.
    const open = live === 'content' && hit.i >= 0 && hit.i === st.rsel
    if (live !== 'geometry' && !open) {
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
        const i = hit.i
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
          now.setCardOpen(true)
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

    if (hit.i >= 0) {
      idx = hit.i
      const r = st.regions[idx]
      if (!r) return
      grip = handle ?? 'move'
      orig = { ...r }
      grab = [px - orig.x, py - orig.y]
      st.select(idx)
    } else if (fresh) {
      // ⌘ is what draws now, and `fresh` is the only way into this branch. A plain
      // press on bare canvas used to draw, which made the canvas a surface where
      // *putting the card away* built a rectangle: the dismissal is a click outside
      // the card, most of the canvas is outside every box, so closing the card left a
      // box behind almost every time. Two gestures make a region — ⌘ and a
      // double-click — and both of them are things you go out of your way to do.
      //
      // Capped, and silently: the cap is the backend's and there is nothing useful to
      // say about it mid-gesture.
      if (st.regions.length >= (st.state?.max_regions ?? 8)) return
      const r = newRegion({ x: px, y: py, w: 0, h: 0 })
      idx = st.regions.length
      grip = 'se'
      orig = { ...r }
      useStore.setState({ regions: [...st.regions, r], rsel: idx })
    } else {
      // Bare canvas, no modifier: the press has already closed the card and that is
      // the whole of what it does.
      return
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
        // **The press that stayed put was a click, and a click opens the box.**
        // Nowhere earlier can tell: the same pointerdown on the same rectangle is the
        // start of a move, of a reshape, and of "show me what is in this one", and
        // which it was is a fact about the *release*. Resolving it here is what lets
        // the card be a thing you ask for without taking the gesture away from
        // framing — press and travel and it never appears, press and let go and it
        // does.
        //
        // Not for a box that was just drawn. ⌘ and a double-click are the two ways to
        // make one, both of them end without travelling, and neither is a request to
        // read what is inside a rectangle that is one second old and empty.
        const drew = grip === 'se' && !orig.w
        const travelled = Math.abs(ev.clientX - e.clientX) + Math.abs(ev.clientY - e.clientY)
        if (!drew && travelled <= CLICK_SLOP) useStore.getState().setCardOpen(true)
        // Selecting is not focusing, and this is the difference between the keyboard
        // working and not. The `preventDefault` above is what owns the drag, and it
        // also suppresses the focus the click would have given the box — so ⌫ and the
        // arrows, which are handled on this layer and find their target with
        // `closest('.rbox')`, only ever reached a box you had *tabbed* to. Clicking one
        // and pressing delete did nothing at all. It matters more now than it did: Tab
        // between boxes is what replaced the map's `‹ 2/5 ›`.
        //
        // The rectangle takes it in every case here, including the click that just
        // opened a card. It used to hand a newly drawn box's caret to `#r-prompt`, on
        // the argument that the card appearing is half the instruction and typing is
        // the other half — but the card no longer appears on its own, so there is no
        // field to hand it to, and while you are framing ⌫ and the arrows are worth
        // more than a caret. Over a render the tap path still does the other thing;
        // that is what `content` is for.
        el?.querySelector<HTMLElement>(`.rbox[data-i="${idx}"]`)?.focus()
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
    const hit = boxAt(stackAt(e.clientX, e.clientY))
    // Said, not swallowed. Without the edit LoRA this used to return in silence, which
    // is indistinguishable from a drop the page never received — and the target is
    // visibly lit, so refusing quietly is a promise made and broken.
    if (hit < 0 && !state?.edit_lora) {
      setRefused(NEED_EDIT_LORA)
      return
    }
    setRefused(null)
    const b64 = await shrinkB64(f)
    if (!b64) return
    if (hit >= 0) {
      attach(hit, 'identity', b64)
      select(hit)
    } else {
      attach('frame', 'scene', b64)
      useStore.getState().setCardOpen(true)
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
         /* The other way to make one, and the reason a plain press no longer does.
            Both gestures that build a region are now deliberate — ⌘ and this — because
            the card's dismissal is "a click anywhere outside it" and most of the canvas
            *is* outside every box: a surface where putting the card away leaves a
            rectangle behind is a surface you cannot put the card away on.
   
            Bare canvas only. Inside a box a double-click is two clicks on the thing
            that opens its card, and ⌘ is already the answer to "a new one, here" where
            there is no bare canvas left to aim at. The press that precedes this one
            returns before `preventDefault`, which is what leaves the double-click to
            arrive at all. */
         onDoubleClick={(e) => {
           if (!onLayer(e)) return
           if ((e.target as HTMLElement).closest('.rins,.rframe-btn')) return
           if (hitAt(e.clientX, e.clientY).i >= 0) return
           const st = useStore.getState()
           if (st.regions.length >= (st.state?.max_regions ?? 8)) return
           const [px, py] = frameXY(e)
           // The size a ⌘-click that never travels already lands on, so the two ways
           // in produce the same rectangle rather than two house styles.
           const [w, h] = [0.28, 0.6]
           const r = newRegion({ x: clamp01(Math.min(px, 1 - w)),
                                 y: clamp01(Math.min(py, 1 - h)), w, h })
           useStore.setState({ regions: [...st.regions, r], rsel: st.regions.length })
           // Over a render the mode a land left behind is `off`, and off paints
           // nothing — so the double-click drew a rectangle nobody could see,
           // selected it, and sent it with the next run, empty sentence and
           // all. Drawing is a request for geometry, so the gesture asks for
           // the mode the way ⌘ does rather than leaving its own box invisible.
           if (over === 'render') useStore.getState().setEdit('geometry')
           // The first box of a session is "start something", not a framing
           // run: the card opening is half the instruction and the caret in
           // its sentence is the other half. Later boxes stay silent — see the
           // pointerup note on `drew` for why a framing run must not be taxed.
           if (!st.regions.length) useStore.getState().setCardOpen(true)
         }}
         /* Hover, resolved by the same rule as the click. Skipped on touch, where a
            move only happens with a finger already down and a box left lit from the
            last tap would be a mark on the picture; and skipped mid-drag, where the
            answer is the box being dragged and nothing else. */
         onPointerMove={(e) => {
           if (e.pointerType === 'touch' || useStore.getState().boxDrag) return
           // Same portal rule as the press: a pointer over the LoRA menu is not a
           // pointer over the picture, and lighting the box underneath it would be
           // the hairline naming something a click cannot reach.
           if (!onLayer(e)) return
           setUnderTo(hitAt(e.clientX, e.clientY).i)
         }}
         onPointerLeave={() => setUnderTo(-1)}
         onDragOver={(e) => {
           if (!onLayer(e)) return
           e.preventDefault()
           // Re-read on every `dragover` rather than latched on enter: a drag can
           // begin outside the layer, and this is the event that repeats.
           //
           // Only the box under the cursor names itself. Eight captions on eight boxes
           // is the same wall of text the per-region rows were removed for, drawn on
           // the picture this time. Which box that is comes off `boxAt`, so a photo
           // lands on the one the caption named — over a small box inside a large one
           // the two used to disagree, and a drop is not a gesture you can take back.
           const hit = boxAt(stackAt(e.clientX, e.clientY))
           setDropHit(hit < 0 ? null : hit)
         }}
         onDragLeave={(e) => {
           if (!e.currentTarget.contains(e.relatedTarget as Node)) setDropHit(null)
         }}
         onDrop={(e) => { if (onLayer(e)) void onDrop(e) }}
         onKeyDown={(e) => {
           // A portal is not the layer here either: Escape inside the LoRA menu is
           // the menu's to answer, and taking it here would close the card out from
           // under the picker that is still open on top of it.
           if (!onLayer(e)) return
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
      {/* The words CLAUDE.md promised the empty canvas — once, where the
          attention already is, and only while there is nothing else to say:
          no boxes, no render under the layer, no file being dragged (the drag
          captions own that moment). One line, because the two gestures land on
          the same rectangle and naming both is a tutorial; double-click is the
          one that needs no modifier key to spell. Gone the moment a first box
          exists, which is the no-tutorial rule's price of admission. */}
      {over === 'frame' && !regions.length && !fileOver && (
        <div className="rl-invite" aria-hidden="true">
          Double-click to place a character
        </div>
      )}
      {regions.map((r, i) => {
        const tag = regionTag(index, r)
        const face = attached(r, 'identity')
        return (
          <div key={r.id} className={['rbox', regionArmed(index, r) ? 'armed' : '',
                                      i === rsel || i === dropHit ? 'sel' : '',
                                      i === under ? 'under' : '',
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
                title="The frame's card — drop a scene to generate inside, an outfit to dress the subjects in, and set how hard every box pushes. Pressed with the card already on the frame, it returns to the first box."
                onClick={() => {
                  select(rsel < 0 ? 0 : -1)
                  // The press that got here closed the card on its way through the
                  // document listener, so this is an open rather than a toggle: the
                  // button is how you *reach* a card, and clicking off it is how you
                  // put it away.
                  useStore.getState().setCardOpen(true)
                }}>
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
