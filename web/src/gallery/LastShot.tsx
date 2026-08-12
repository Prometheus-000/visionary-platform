import { fileUrl } from '../api/routes'
import type { GalleryItem } from './types'

/**
 * The way back to what you made is a picture of what you made.
 *
 * The Camera app's move: an icon says "gallery", which you have to already want; a thumbnail
 * says "this is the last thing you rendered", which is what you are reaching for. It sits
 * beside Generate because those two are one loop — you press one and then want the other.
 * Putting it in the top bar meant pressing Generate at the bottom of a 1194px screen and then
 * reaching to the opposite corner to look at the result: the hand crossing the whole device
 * and coming back, for two things that belong within a thumb's travel of each other.
 *
 * **`.shot-back` is declared inside a media query, and that is the point.** This was declared
 * outside one for a while, so it sat beside Generate at 1512px too — where the drawer is
 * already a column next to the picture and the header button already opens it, making this a
 * second door onto a room that has one. Two changes made for a phone reached desktop that way;
 * both were wrong there.
 */
export function LastShot({ items, onOpen }: {
  items: GalleryItem[]
  onOpen: (rows: GalleryItem[], i: number) => void
}) {
  const it = items[0]
  // Falls back to nothing rather than to a symbol: with an empty volume there is no last
  // generation, and a hole where a picture goes is worse than a symbol only if there is
  // something to put in it.
  if (!it?.files[0]) return null
  const src = fileUrl(it.job_id, it.files[0])
  return (
    <button className="ico shot-back" title="Last generation" type="button"
            onClick={() => onOpen(items, 0)}>
      {it.kind === 'video'
        ? <video src={`${src}#t=0.04`} preload="metadata" muted playsInline />
        : <img src={src} alt="" />}
    </button>
  )
}
