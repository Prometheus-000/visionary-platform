import { useCallback, useEffect, useRef, useState } from 'react'

import { coverUrl, type GalleryItem } from './types'

/**
 * The picture, full screen, and the gesture that pages between them.
 *
 * Everything about the drag is imperative — refs and direct transform writes,
 * with React seeing nothing until the slide lands. That is not an optimisation,
 * it is the difference between this working and not: a `setState` per
 * `pointermove` re-renders three slides, two of which hold decoded images or a
 * playing `<video>`, at pointer rate.
 *
 * Three slides, not one. The middle is what you are looking at and the
 * neighbours are already in the DOM, because the drag has to show them: a page
 * that only swaps on release is a cut, and a cut does not tell you which way
 * you went or that there is anything either side. This is the one place in the
 * app where an animation carries information rather than decorating.
 */
export function Viewer({
  items,
  index,
  onIndex,
  onClose,
  onAll,
}: {
  items: GalleryItem[]
  index: number
  onIndex: (i: number) => void
  onClose: () => void
  onAll: () => void
}) {
  const elRef = useRef<HTMLDivElement>(null)
  const trackRef = useRef<HTMLDivElement>(null)
  /** Chrome off. A tap on the picture asks for more of it. */
  const [bare, setBare] = useState(false)

  const many = items.length > 1
  const it = items[index]

  // The gesture's whole state. None of it belongs to React — it changes at
  // pointer rate and nothing renders from it.
  const g = useRef({ sx: 0, sy: 0, st: 0, w: 1, down: false, dragging: false, justDragged: false })

  /**
   * Stops at the ends rather than wrapping. A set that loops has no edges, and
   * "am I back where I started" is the question you cannot answer while
   * flicking.
   */
  const step = useCallback(
    (d: number) => {
      const i = index + d
      if (!items.length || i < 0 || i >= items.length) return
      onIndex(i)
    },
    [index, items.length, onIndex],
  )

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') return onClose()
      if (e.key === 'ArrowLeft') return step(-1)
      if (e.key === 'ArrowRight') return step(1)
    }
    document.addEventListener('keydown', onKey)
    return () => document.removeEventListener('keydown', onKey)
  }, [onClose, step])

  // The track is re-centred whenever the index changes, without a transition,
  // so the rebuilt neighbours are already in place before anything is shown.
  useEffect(() => {
    const track = trackRef.current
    if (!track || !many) return
    track.classList.remove('snap')
    track.style.transform = 'translateX(-100%)'
  }, [index, many])

  const settle = (dx: number, commit: boolean) => {
    const track = trackRef.current
    if (!track) return
    track.classList.add('snap')
    track.style.transform = `translateX(${commit ? (dx < 0 ? '-200%' : '0%') : '-100%'})`
  }

  const onPointerDown = (e: React.PointerEvent) => {
    if (!many || (e.target as HTMLElement).closest('button')) return
    const s = g.current
    s.sx = e.clientX
    s.sy = e.clientY
    s.st = e.timeStamp
    s.down = true
    s.dragging = false
    s.w = elRef.current?.getBoundingClientRect().width || 1
    trackRef.current?.classList.remove('snap')
    // Captured, so a drag that wanders off the picture — or off the window —
    // keeps arriving here instead of stopping wherever the pointer went.
    try {
      elRef.current?.setPointerCapture(e.pointerId)
    } catch {
      /* a pointer that has already gone is not an error worth having */
    }
  }

  const onPointerMove = (e: React.PointerEvent) => {
    const s = g.current
    if (!s.down) return
    const dx = e.clientX - s.sx
    const dy = e.clientY - s.sy
    // Committed to horizontal only once it is clearly horizontal, so a vertical
    // flick on a tall picture is still a scroll and not a half-page.
    if (!s.dragging) {
      if (Math.abs(dx) < 10 || Math.abs(dx) <= Math.abs(dy)) return
      s.dragging = true
    }
    e.preventDefault()
    // Resistance at the ends: it moves, so the gesture is alive, but a third as
    // far, which is how an edge is felt rather than announced.
    const edge = (dx > 0 && index <= 0) || (dx < 0 && index >= items.length - 1)
    const track = trackRef.current
    if (track) track.style.transform = `translateX(calc(-100% + ${dx * (edge ? 0.3 : 1)}px))`
  }

  const release = (e: React.PointerEvent) => {
    const s = g.current
    if (!s.down) return
    const dx = e.clientX - s.sx
    const was = s.dragging
    s.down = false
    s.dragging = false
    if (!was) return

    // A drag ends in a click. Without this the click lands on the backdrop and
    // closes the viewer, so every successful swipe shut the thing it was
    // paging — and on a trackpad that is the only outcome you ever see.
    s.justDragged = true
    setTimeout(() => {
      s.justDragged = false
    }, 0)

    // Distance or speed, not distance alone. A quarter of the width is a long
    // way on a 1200px viewer, and requiring it made every quick flick fall
    // back — the gesture people actually use to go through twenty takes is a
    // short fast one. 0.45px/ms is about the speed of a flick that means it.
    const v = Math.abs(dx) / Math.max(1, e.timeStamp - s.st)
    const meant = Math.abs(dx) > s.w * 0.25 || (v > 0.45 && Math.abs(dx) > 36)
    const commit =
      meant && !((dx > 0 && index <= 0) || (dx < 0 && index >= items.length - 1))
    settle(dx, commit)
    if (!commit) return

    // Swapped after the slide lands, so the rebuild is invisible: the picture
    // already sits where the new middle slide will put it.
    const next = index + (dx < 0 ? 1 : -1)
    trackRef.current?.addEventListener('transitionend', () => onIndex(next), { once: true })
  }

  const onClick = (e: React.MouseEvent) => {
    if (g.current.justDragged) return
    const t = e.target as HTMLElement
    if (t.closest('.x')) return onClose()
    if (t.closest('button,video')) return // controls keep their own jobs
    // On the picture, a click asks for more of it — chrome off. Off the picture
    // is where the two devices differ, and they should: clicking the dark
    // surround to dismiss is a desktop convention old enough that removing it
    // reads as a broken dialog, while on glass the picture fills the screen and
    // that surround is a few pixels nobody aims at. So the mouse keeps its
    // click-away and touch does not have to grow one.
    if (t.closest('.lb-slide img,.lb-slide video')) return setBare((b) => !b)
    if (matchMedia('(hover:hover)').matches) return onClose()
    setBare((b) => !b)
  }

  const slide = (i: number) => {
    const s = items[i]
    if (!s) return <div className="lb-slide" key={`empty${i}`} />
    const u = coverUrl(s)
    return (
      <div className="lb-slide" key={`${s.job_id}:${i}`}>
        {s.kind === 'video' ? (
          // draggable={false} because an <img>/<video> is natively draggable:
          // mousedown and move starts an HTML drag, the browser fires
          // pointercancel, and the gesture dies one frame in. Touch never takes
          // that path, so the same code tracked a thumb and refused a trackpad.
          <video src={u} controls loop playsInline draggable={false} />
        ) : (
          <img src={u} alt="" draggable={false} />
        )}
      </div>
    )
  }

  if (!it) return null

  return (
    <div
      className={`lb${bare ? ' bare' : ''}`}
      ref={elRef}
      onPointerDown={onPointerDown}
      onPointerMove={onPointerMove}
      onPointerUp={release}
      // Cancel means something took the gesture away, which is not the same as
      // you finishing it — so it goes back rather than committing to a take you
      // may never have asked for.
      onPointerCancel={() => {
        if (!g.current.down) return
        g.current.down = false
        g.current.dragging = false
        settle(0, false)
      }}
      onClick={onClick}
    >
      <button className="x" title="Close" type="button">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.1"
             strokeLinecap="round" strokeLinejoin="round">
          <path d="M9 6l6 6-6 6" />
        </svg>
      </button>

      {many && (
        <>
          <button className="lb-nav prev" type="button" disabled={index <= 0}
                  onClick={() => step(-1)}>
            <Chevron />
          </button>
          <button className="lb-nav next" type="button" disabled={index >= items.length - 1}
                  onClick={() => step(1)}>
            <Chevron />
          </button>
          <span className="lb-at">
            {index + 1} / {items.length}
          </span>
        </>
      )}

      <button className="lb-all" title="All generations" type="button" onClick={onAll}>
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.9"
             strokeLinejoin="round">
          <rect x="3" y="3" width="7.5" height="7.5" rx="1.4" />
          <rect x="13.5" y="3" width="7.5" height="7.5" rx="1.4" />
          <rect x="3" y="13.5" width="7.5" height="7.5" rx="1.4" />
          <rect x="13.5" y="13.5" width="7.5" height="7.5" rx="1.4" />
        </svg>
      </button>

      <div className="lb-track" ref={trackRef}
           style={many ? { transform: 'translateX(-100%)' } : undefined}>
        {many ? [slide(index - 1), slide(index), slide(index + 1)] : slide(index)}
      </div>
    </div>
  )
}

function Chevron() {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.1"
         strokeLinecap="round" strokeLinejoin="round">
      <path d="M15 6l-6 6 6 6" />
    </svg>
  )
}
