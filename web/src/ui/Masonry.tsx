import { useLayoutEffect, useRef, useState, type ReactNode } from 'react'

/**
 * A grid whose cells are the shape of the pictures in them.
 *
 * The thing it replaces is a letterbox. Every card was `aspect-ratio:4/3` with
 * `object-fit:contain`, so a portrait still arrived as a small picture between
 * two bars of `--wash-2` — the grid was tidy and the pictures were smaller than
 * the space spent on them. `contain` was always the right half of that decision
 * and the fixed box was the half that cost; this drops the box instead.
 *
 * **Not a dependency, and the reason is order rather than size.** Every cheap
 * way to do this reflows the reading order:
 *
 *  - CSS `columns` fills a column top to bottom, so a listing that is newest
 *    first becomes newest *down the left edge*. The gallery's whole contract is
 *    that the top-left card is the last thing you made.
 *  - `react-masonry-css` and the rest distribute by `i % cols`, which keeps the
 *    order and gives up the packing — the columns end up as uneven as the
 *    pictures are.
 *  - The libraries that do pack properly measure the DOM, so they reflow when
 *    each image decodes. We do not need to measure anything: `/api/gallery`
 *    and `/api/dataset` both carry the pixel dimensions of every item, so the
 *    layout is known before a single byte of picture is fetched.
 *  - Native `grid-template-rows:masonry` is still moving under the spec (it is
 *    being re-argued as `item-flow`), so it is not something to ship on yet.
 *
 * What is left is a greedy shortest-column pack, which is twenty lines and has
 * a property none of the above has: with every column starting empty, the first
 * row *is* items 1..n left to right, and each item after that lands in the
 * column that is currently shortest — so it reads approximately row-major
 * rather than column-major. Chronology survives.
 *
 * It is also stable under paging. Greedy over a prefix is the prefix of greedy
 * over the whole list, so appending a page never moves a card already on
 * screen — which an incremental measure-and-place cannot promise.
 *
 * The gap is a prop rather than a class because the same number is used twice:
 * once to lay the columns out and once to add up the heights they are packed
 * by. A stylesheet gap and a JS gap that drift are a packer optimising for a
 * layout that is not on screen.
 */
export function Masonry<T>({
  id, className, items, render, ratio, cols: fixed, min = 232, gap = 14, chrome = 0,
}: {
  id?: string
  className?: string
  items: T[]
  render: (item: T) => ReactNode
  /** Height over width. The fallback belongs to the caller: what an item with
   *  no recorded size should be *drawn* as is the same question as what it
   *  should be packed as, and two answers would disagree visibly. */
  ratio: (item: T) => number
  /** Column count, when something else already decides it — the dataset
   *  editor's density slider is a person saying how big they want the tiles. */
  cols?: number
  /** Narrowest a column may be before one is dropped. Ignored under `cols`. */
  min?: number
  gap?: number
  /** Per-item height that is not the picture: the card's footer, the tile's
   *  caption box. Constant per surface, and left out entirely it packs a tall
   *  column of short cards as if the chrome were free. */
  chrome?: number
}) {
  const box = useRef<HTMLDivElement>(null)
  const [width, setWidth] = useState(0)

  // Layout, not effect: the first pass renders with width 0 and would paint a
  // single column for a frame if the measurement waited for the browser.
  useLayoutEffect(() => {
    const el = box.current
    if (!el) return
    setWidth(el.clientWidth)
    const ro = new ResizeObserver(() => setWidth(el.clientWidth))
    ro.observe(el)
    return () => ro.disconnect()
  }, [])

  const n = Math.max(1, fixed ?? Math.floor((width + gap) / (min + gap)))
  const colW = (width - gap * (n - 1)) / n

  const columns: T[][] = Array.from({ length: n }, () => [])
  const heights = new Array(n).fill(0)
  for (const item of items) {
    let c = 0
    for (let i = 1; i < n; i++) if (heights[i] < heights[c]) c = i
    columns[c]!.push(item)
    heights[c] += colW * ratio(item) + chrome + gap
  }

  // `align-items:flex-start` is what stops a short column stretching to the
  // tallest one and spreading its own cards out to fill it — which looks like
  // the packer failed and is really the columns being equal height by default.
  return (
    <div id={id} className={`masonry${className ? ` ${className}` : ''}`} ref={box} style={{ gap }}>
      {columns.map((col, i) => (
        <div className="mcol" key={i} style={{ gap }}>{col.map(render)}</div>
      ))}
    </div>
  )
}
