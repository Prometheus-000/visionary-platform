import { ago, coverUrl, type GalleryItem } from './types'
import { IconClose, IconDownload, IconMore, IconPhoto, IconPlay } from '../icons'

/**
 * One generation in a grid.
 *
 * The cover is `contain`, never `cover`, on desktop — a ragged grid is the
 * right trade because cropping throws away the thing you opened the gallery to
 * see. The stylesheet handles that; the only reason it is worth repeating here
 * is that "tidy the grid" is a change somebody will want to make.
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
  const src = coverUrl(item)
  const extra = item.files.length > 1 ? ` · ${item.files.length}` : ''

  const stop = (fn: () => void) => (e: React.MouseEvent) => {
    e.stopPropagation()
    fn()
  }

  return (
    <div className="gal">
      {item.kind === 'video' ? (
        // A poster frame would be a second request per card, so the video
        // loads metadata only. But metadata is dimensions and duration, not a
        // picture: with nothing to paint, every clip was a black rectangle you
        // had to open to identify. `#t=` is the media fragment that fixes it —
        // the browser seeks there and paints the frame out of bytes it already
        // has, so the card costs no extra request. Not 0: seeking to exactly
        // zero is not required to decode a frame, and some browsers leave the
        // canvas blank.
        <video className="media" src={`${src}#t=0.04`} preload="metadata" muted
               playsInline onClick={onOpen} />
      ) : (
        <img className="media" src={src} alt="" loading="lazy" onClick={onOpen} />
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
