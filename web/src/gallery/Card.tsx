import { useRef } from 'react'

import { ago, coverOf, fullUrl, type GalleryItem } from './types'
import { IconMore, IconPhoto, IconPlay } from '../icons'
import { useNearViewport } from '../media/inview'
import { Thumb } from '../media/thumb'

/**
 * One generation in a grid.
 *
 * The cover is `contain`, never `cover`, on desktop — a ragged grid is the
 * right trade because cropping throws away the thing you opened the gallery to
 * see. The stylesheet handles that; the only reason it is worth repeating here
 * is that "tidy the grid" is a change somebody will want to make.
 *
 * `contain` inside a fixed 4:3 box is what that used to mean, and the bars it
 * left were the cost of keeping the grid square-shouldered. On desktop the box
 * is now the picture's own shape and the cards are packed into columns, so
 * nothing is letterboxed and nothing is cropped — see `Masonry`. Below 1024px
 * the square crop stays, deliberately.
 *
 * The picture is `/api/cover`, not `/api/file`: see `coverOf`. The viewer gets the
 * original, which is where full resolution is the whole point.
 *
 * There is no hover cluster over the picture any more. Download and Delete sat there and
 * again in the ⋯ menu underneath — the only actions in the app with two doors, and the
 * two doors disagreed about what a card can do: the cluster held two verbs, the menu
 * holds six, including reuse, which is the reason the sidecar is kept at all. One home,
 * and it is the menu, because that is the one that can hold the whole set. Opening is
 * untouched by this and never lived in the cluster — it has always been a click on the
 * picture itself.
 */
export function Card({
  item,
  onOpen,
  onMenu,
  busy,
  aspect,
}: {
  item: GalleryItem
  onOpen: () => void
  onMenu: (anchor: HTMLElement) => void
  /** The picture's own shape, `w/h`, when the card is being packed by one. Handed
   *  down rather than read from the item here because the two homes that are not
   *  the masonry want the stylesheet's box: the drawer is a single column, and
   *  below 1024px the grid crops to squares on purpose. A card that decided this
   *  for itself would break both from the inside. */
  aspect?: string | null
  /** This card's own delete is out. Handed down rather than inferred here because the
   *  request belongs to the gallery, which is also what reloads the listing after it. */
  busy?: boolean
}) {
  const box = useRef<HTMLDivElement>(null)
  const isVideo = item.kind === 'video'
  const near = useNearViewport(box, isVideo)

  const extra = item.files.length > 1 ? ` · ${item.files.length}` : ''

  // Dimmed and inert while its own delete is out. The menu closes on the click that starts
  // one, so from then until the new listing lands the only thing on screen was a card that
  // still looked exactly like a card you could delete — and the second confirm dialog that
  // invited was for a file already on its way out. Inline rather than a class because it
  // is two declarations and the stylesheet has no `.gal` state to hang them on.
  const pending = busy ? { opacity: 0.5, pointerEvents: 'none' as const } : undefined

  // Overrides `.gal .media`'s 4/3, and only ever downward in specificity terms:
  // an item with no recorded size passes nothing and keeps the box.
  const shape = aspect ? { aspectRatio: aspect } : undefined

  return (
    <div className="gal" ref={box} aria-busy={busy || undefined} style={pending}>
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
          ? <video className="media" style={shape} src={`${fullUrl(item)}#t=0.04`}
                   preload="metadata" muted playsInline onClick={onOpen} />
          : <div className="media" style={shape} onClick={onOpen} />
      ) : (
        // Not a bare <img>: the Thumb queues the fetch — a page of covers
        // fired at once is what starved the status poll during a run — and a
        // failure retries with backoff before it is allowed to stay blank,
        // where an <img> whose one request failed was blank forever.
        <Thumb className="media" style={shape} url={coverOf(item)} onClick={onOpen} />
      )}

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
