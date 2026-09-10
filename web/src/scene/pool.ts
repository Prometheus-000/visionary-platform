import { shrinkB64, toB64 } from '../media/files'
import type { Media, PoolFile } from './model'

/**
 * A dropped file, ready to be pointed at.
 *
 * **One pool, keyed by content.** The same photograph in two buckets is one
 * upload and one encode — two characters cut from the same still, or one picture
 * that is both the wardrobe and the body reference. Keyed by a hash rather than a
 * name because a name is the one property two copies of a file are least likely
 * to share: dragged from two folders, exported twice, renamed on the way in.
 *
 * It also makes `<Picture N>` derivable. Numbering walks the cast collecting
 * distinct ids, so removing a bucket renumbers correctly and the two-parallel-
 * arrays invariant `refs`/`refRoles` had to keep by hand is not fixed but
 * *unrepresentable*.
 */

/** What a browser MIME type means to this app. */
export function mediaOf(type: string): Media | null {
  if (type.startsWith('image/')) return 'image'
  if (type.startsWith('video/')) return 'video'
  if (type.startsWith('audio/')) return 'audio'
  return null
}

/**
 * The id, hashed over **what travels** rather than over the file on disk.
 *
 * Two pictures that shrink to the same bytes are one upload, which is the thing
 * dedupe is protecting — the request, not the filesystem. Hashing the original
 * would make a photograph and its own re-export two entries in `references[]`
 * pointing at one picture, and every `<Picture N>` after the first would be off
 * by one for no visible reason.
 */
async function idOf(b64: string): Promise<string> {
  const bytes = new TextEncoder().encode(b64)
  const digest = await crypto.subtle.digest('SHA-256', bytes)
  return [...new Uint8Array(digest).slice(0, 12)]
    .map((n) => n.toString(16).padStart(2, '0')).join('')
}

/**
 * Read a file into the pool's shape, or null if the browser could not.
 *
 * Images go through the same shrink a region photo does — nine of them in one
 * JSON body is H3's own maximum and therefore the case to size for, not the
 * unlucky one. Video and audio are sent whole: there is nothing here that can
 * re-encode either.
 */
export async function intake(
  f: File,
  from?: { handle: string; file: string },
): Promise<PoolFile | null> {
  const kind = mediaOf(f.type)
  if (!kind) return null
  const b64 = await (kind === 'image' ? shrinkB64(f) : toB64(f))
  // A file the browser could not decode. Kept out rather than stored empty: a
  // slot holding a thumbnail-less entry would send the string "null" to the GPU
  // and fail there instead of here, where the file it came from is still in hand.
  if (!b64) return null
  return {
    id: await idOf(b64),
    name: f.name,
    kind,
    b64,
    // Revoked when the entry leaves the pool — see `dropFile`. Not `dataUrl(b64)`:
    // a thumbnail for every reference as a data URI is the whole payload rendered
    // twice, once for the model and once for a 64px square.
    url: URL.createObjectURL(f),
    // Only a recall passes this. See `PoolFile.from` and `refreshArsenal`.
    ...(from ? { from } : {}),
  }
}
