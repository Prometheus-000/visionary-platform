import { useEffect, useRef, useState } from 'react'

import { ago, coverOf, fullUrl, type GalleryItem } from './types'
import { IconClose, IconDownload, IconMore, IconPhoto, IconPlay } from '../icons'

/**
 * Once the card is near the viewport, and never false again.
 *
 * Only clips need this — an `<img>` has `loading="lazy"` and the browser does the same
 * job for free. A `<video preload="metadata">` has no equivalent, so every clip in the
 * grid issued a range request on mount whether or not it was on screen: forty clips,
 * forty requests, before you had scrolled to the second row. Each one is a
 * `FileResponse` holding a descriptor on the volume, which is what refuses
 * `volume.reload()` and freezes the listing the grid is trying to show.
 *
 * It latches rather than tracking visibility. Unsetting `src` on scroll-away would
 * re-fetch on scroll-back, which turns a bounded cost into an unbounded one — the
 * opposite of the point.
 */
function useNearViewport(ref: React.RefObject<HTMLElement | null>, on: boolean): boolean {
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

/**
 * One generation in a grid.
 *
 * The cover is `contain`, never `cover`, on desktop — a ragged grid is the
 * right trade because cropping throws away the thing you opened the gallery to
 * see. The stylesheet handles that; the only reason it is worth repeating here
 * is that "tidy the grid" is a change somebody will want to make.
 *
 * The picture is `/api/cover`, not `/api/file`: see `coverOf`. The viewer gets the
 * original, which is where full resolution is the whole point.
 */
export function Card({
  item,
  onOpen,
  onDownload,
  onDelete,
  onMenu,
}: {
  item: GalleryItem
  onOpen: () => void
  onDownload: () => void
  onDelete: () => void
  onMenu: (anchor: HTMLElement) => void
}) {
  const box = useRef<HTMLDivElement>(null)
  const isVideo = item.kind === 'video'
  const near = useNearViewport(box, isVideo)

  // A cover that lost the race with the mount 404s, and re-fetching the listing does
  // not fix it: React re-renders this card with the same key and the same src, and the
  // browser does not retry a failed image. One retry, once, with a buster so it is not
  // answered out of the HTTP cache; a second failure paints nothing, as before.
  const retried = useRef(false)
  const [bust, setBust] = useState('')
  const onError = () => {
    if (retried.current) return
    retried.current = true
    setBust(`?r=${performance.now().toFixed(0)}`)
  }

  const extra = item.files.length > 1 ? ` · ${item.files.length}` : ''

  const stop = (fn: () => void) => (e: React.MouseEvent) => {
    e.stopPropagation()
    fn()
  }

  return (
    <div className="gal" ref={box}>
      {isVideo ? (
        // No cover route for a clip — web_image has no ffmpeg, and the card would
        // rather fetch the frame it needs than have the server ship a whole mp4 to
        // make a picture of it.
        //
        // A poster frame would be a second request per card, so the video
        // loads metadata only. But metadata is dimensions and duration, not a
        // picture: with nothing to paint, every clip was a black rectangle you
        // had to open to identify. `#t=` is the media fragment that fixes it —
        // the browser seeks there and paints the frame out of bytes it already
        // has, so the card costs no extra request. Not 0: seeking to exactly
        // zero is not required to decode a frame, and some browsers leave the
        // canvas blank.
        //
        // Gated, because that request is unconditional and forty of them at once is
        // what jams the volume — see `useNearViewport`.
        near
          ? <video className="media" src={`${fullUrl(item)}#t=0.04`} preload="metadata"
                   muted playsInline onClick={onOpen} />
          : <div className="media" onClick={onOpen} />
      ) : (
        <img className="media" src={`${coverOf(item)}${bust}`} alt="" loading="lazy"
             onError={onError} onClick={onOpen} />
      )}

      <div className="quick">
        <button title="Download" type="button" onClick={stop(onDownload)}>
          <IconDownload />
        </button>
        <button title="Delete" type="button" onClick={stop(onDelete)}>
          <IconClose />
        </button>
      </div>

      <div className="foot">
        <span className="kind">{item.kind === 'video' ? <IconPlay /> : <IconPhoto />}</span>
        <span className="when">
          {ago(item.created ?? item.modified)}
          {extra}
        </span>
        <span className="grow" />
        <button className="more" title="More" type="button"
                onClick={(e) => {
                  e.stopPropagation()
                  onMenu(e.currentTarget)
                }}>
          <IconMore />
        </button>
      </div>
    </div>
  )
}
