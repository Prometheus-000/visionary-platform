import { useEffect, useRef, useState } from 'react'

import { dataUrl } from '../media/files'
import { readSize } from '../console/size'
import { useStore } from '../store'
import { RegionLayer } from './RegionLayer'

/**
 * The frame regional mode draws on: the render's own aspect, at the size the canvas
 * can give it.
 *
 * **Sized off the canvas, never off the viewport.** The console under it grows and
 * shrinks with what is open, so a `dvh` sum is wrong the moment a popover opens.
 * Same measurement `layoutShots` does and for the same reason, and every subtrahend
 * is measured rather than guessed: `clientHeight` includes the canvas padding, and
 * the caption below the grid is a real element with a real height.
 *
 * Both dimensions in pixels. `aspect-ratio` with an auto width is the obvious way to
 * write this and does not work here: the frame is a flex item in a column, so its
 * width is the cross size and the ratio never transferred into it — the box came out
 * 2px wide, which is exactly its two borders and nothing else.
 */
export function Frame() {
  const img = useStore((s) => s.img)
  const plate = useStore((s) => s.plate)
  const el = useRef<HTMLDivElement>(null)
  const [box, setBox] = useState<{ w: number; h: number } | null>(null)

  const [rw, rh] = readSize(img)
  const ar = rw && rh ? rw / rh : 1

  useEffect(() => {
    const frame = el.current
    const canvas = frame?.closest<HTMLElement>('#canvas')
    if (!frame || !canvas) return
    const fit = () => {
      const cs = getComputedStyle(canvas)
      const cap = canvas.querySelector<HTMLElement>('#gen-meta')
      const availH = canvas.clientHeight - parseFloat(cs.paddingTop)
        - parseFloat(cs.paddingBottom) - (cap?.offsetHeight ?? 0) - 12
      const availW = canvas.clientWidth - parseFloat(cs.paddingLeft) - parseFloat(cs.paddingRight)
      // A container with no width has not been laid out yet — a hidden view, a tab
      // restored in the background. Writing that measurement through would set the
      // frame to 0×0 and every drag on it would divide by zero; keeping the last
      // real numbers costs nothing, because the observer fires again the moment it
      // does have a size.
      if (availW <= 0) return
      // A floor, because a console with everything open can leave less height than
      // a frame you could actually drag inside — at that point letting the canvas
      // scroll beats offering a 40px target.
      let h = Math.max(160, Math.min(availH, availW / ar))
      let w = h * ar
      if (w > availW) {
        w = availW
        h = w / ar
      }
      setBox({ w: Math.round(w), h: Math.round(h) })
    }
    fit()
    const ro = new ResizeObserver(fit)
    ro.observe(canvas)
    return () => ro.disconnect()
  }, [ar])

  return (
    <div id="frame" ref={el} className="frame can-drop"
         style={{ ['--frame-w' as string]: `${box?.w ?? 0}px`,
                  ['--frame-h' as string]: `${box?.h ?? 0}px` }}>
      {/* A frame you can see the background in is the difference between placing
          people in a room and placing rectangles in a void. */}
      {plate.scene && <img className="plate" src={dataUrl(plate.scene)} alt="" />}
      {/* Thirds are a viewfinder for an empty frame; over a photograph they are just
          lines on someone's picture. */}
      {!plate.scene && <div className="thirds" />}
      <RegionLayer />
    </div>
  )
}
