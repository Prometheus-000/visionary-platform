/**
 * A region is a rectangle on the canvas, so it is drawn on the canvas.
 *
 * The four coordinates were the right parameter and the wrong primary: "0.5 0 0.5 1"
 * is a rectangle you rebuild in your head every time, which is why the row that
 * carried those numbers also had to carry a 32px picture of them. That picture is
 * the frame now, at size and grabbable — and the numbers, still in the inspector,
 * are a readout that moves while you drag. Dragging is what teaches them; they never
 * taught the dragging.
 */
import { attached, type Region } from '../store'
import { stripLoras, type LoraFile } from '../lora/tokens'

/** Below this a box is not grabbable, and 0 is rejected by the backend outright. */
export const MIN_SIDE = 0.04
/** Halves, thirds and quarters. Snapping to these is what makes an even split a
 *  gesture rather than a menu. */
export const SNAP_TO = [0, 1 / 4, 1 / 3, 1 / 2, 2 / 3, 3 / 4, 1]
/** A fraction of the frame, so it feels the same at any size. */
export const SNAP_EPS = 0.015

export const clamp01 = (v: number) => Math.max(0, Math.min(1, Number.isFinite(v) ? v : 0))

/**
 * Snap to the frame's own landmarks and to the other boxes' edges. Alt turns it off
 * for the times when the tidy answer is the wrong one.
 *
 * Named `snapEdge` and not `snap` because the size control already owns that name
 * for rounding pixels to the VAE grid — two of them in one module scope is a page
 * that does not load at all.
 */
export function snapEdge(
  v: number,
  axis: 'x' | 'y',
  regions: Region[],
  skip: number,
  alt: boolean,
): number {
  if (alt) return v
  let best = v
  let dist = SNAP_EPS
  const cands = SNAP_TO.slice()
  regions.forEach((r, i) => {
    if (i === skip) return
    cands.push(axis === 'x' ? r.x : r.y, axis === 'x' ? r.x + r.w : r.y + r.h)
  })
  for (const c of cands) {
    const d = Math.abs(v - c)
    if (d < dist) {
      dist = d
      best = c
    }
  }
  return best
}

/** A box is "armed" when it holds an identity: a LoRA name that resolves to a file,
 *  or a photograph. That is the distinction that decides what comes out — an empty
 *  rectangle is filled by the scene prompt, a box with an identity in it is a
 *  person — and it is the same one the old 32px plots drew. */
export const regionArmed = (_index: LoraFile[], r: Region): boolean =>
  !!(attached(r, 'identity') || r.lora)

/** What the box calls itself. A name you gave the character wins over everything,
 *  because it is the one label somebody chose rather than derived — and it is the
 *  handle the Arsenal knows them by, so the box reads as the person rather than as
 *  the file. Then the LoRA name, then the prompt: a photo-only box still has words
 *  worth showing. */
export function regionTag(_index: LoraFile[], r: Region): { text: string; muted: boolean } {
  if (r.name) return { text: `@${r.name}`, muted: false }
  if (r.lora) return { text: r.lora.rel, muted: false }
  const words = stripLoras(r.prompt || '').trim()
  if (words) return { text: words.length > 28 ? `${words.slice(0, 27)}…` : words, muted: false }
  return attached(r, 'identity') ? { text: 'photo', muted: true } : { text: '', muted: false }
}

/** Overwrites, unlike the Columns/Rows select it replaces: that filled only the
 *  coordinates nobody had typed into, and once boxes are drawn there is no such
 *  thing as an untouched coordinate. */
export function distribute(regions: Region[], cols: boolean): Region[] {
  const n = regions.length
  if (!n) return regions
  return regions.map((r, i) => ({
    ...r,
    x: cols ? i / n : 0,
    y: cols ? 0 : i / n,
    w: cols ? 1 / n : 1,
    h: cols ? 1 : 1 / n,
  }))
}

/**
 * The stack, in the shape `/api/generate` takes.
 *
 * A box with no words, no LoRA and no photograph is an empty rectangle, and dropping
 * it *here* is safe in a way filtering server-side would not be: the boxes and the
 * rows are built from this one list in this one order, so they stay paired by index
 * whatever comes out.
 *
 * The second LoRA in a box is sent rather than dropped, so the backend can say "a
 * region takes one" about it instead of this quietly losing it.
 */
export function readRegions(_index: LoraFile[], regions: Region[], on: boolean) {
  if (!on) return []
  return regions
    .map((r) => {
      return {
        prompt: stripLoras(r.prompt || ''),
        // Still a list, because that is the shape the backend takes — the node
        // itself allows one per box, which is why the card offers one.
        loras: r.lora ? [{ path: r.lora.path, unet: r.lora.strength }] : [],
        ref: attached(r, 'identity'),
        x: r.x, y: r.y, width: r.w, height: r.h,
      }
    })
    .filter((r) => r.prompt || r.loras.length || r.ref)
}
