import { useEffect, useState } from 'react'

/**
 * Once the element is near the viewport, and never false again.
 *
 * Only clips need this — an `<img>` has `loading="lazy"` and the browser does
 * the same job for free. A `<video preload="metadata">` has no equivalent, so
 * every clip in a grid issues a range request on mount whether or not it is on
 * screen: forty clips, forty requests, before you have scrolled to the second
 * row. Each one is a `FileResponse` holding a descriptor on the volume, which
 * is what refuses `volume.reload()` and freezes the listing the grid is trying
 * to show.
 *
 * It latches rather than tracking visibility. Unsetting `src` on scroll-away
 * would re-fetch on scroll-back, which turns a bounded cost into an unbounded
 * one — the opposite of the point.
 *
 * Shared by the gallery card and the dataset tile. It lived inside the gallery
 * card until a dataset could hold clips too; a second copy would have been a
 * second place for the latch to be got wrong, and the fault it prevents is
 * invisible until the volume freezes.
 */
export function useNearViewport(ref: React.RefObject<HTMLElement | null>, on: boolean): boolean {
  const [near, setNear] = useState(false)
  useEffect(() => {
    if (!on || near || !ref.current) return
    // No IntersectionObserver is not a reason to show nothing: fall through to
    // loading, which is exactly the behaviour this replaces.
    if (typeof IntersectionObserver === 'undefined') return setNear(true)
    const io = new IntersectionObserver(
      (es) => es.some((e) => e.isIntersecting) && setNear(true),
      // A screen of margin, so the clip is decoded by the time it is scrolled to
      // rather than starting to load as it arrives.
      { rootMargin: '200px' },
    )
    io.observe(ref.current)
    return () => io.disconnect()
  }, [on, near, ref])
  return near
}
