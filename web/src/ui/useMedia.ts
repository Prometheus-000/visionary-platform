import { useEffect, useState } from 'react'

/**
 * A media query as a boolean, for the splits that cannot be made in CSS.
 *
 * Almost every one here is made in the stylesheet and should stay there. This
 * exists for the gallery, where the two layouts are different *DOM*: below
 * 1024px the grid crops to squares and reads row-major, and a masonry's column
 * elements would put that grid in chronological order down each column. There
 * is no CSS that unpicks that — `display:contents` on the columns flattens
 * them into the grid in column order, which is the wrong order rather than no
 * order.
 *
 * `matchMedia` rather than a width in state, so the listener fires on the
 * crossing rather than on every pixel of a drag.
 */
export function useMedia(query: string): boolean {
  const [on, setOn] = useState(() => matchMedia(query).matches)
  useEffect(() => {
    const mq = matchMedia(query)
    const sync = () => setOn(mq.matches)
    sync()
    mq.addEventListener('change', sync)
    return () => mq.removeEventListener('change', sync)
  }, [query])
  return on
}
