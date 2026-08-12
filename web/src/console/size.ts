/**
 * The ratio picker and the two pixel boxes are one control, not two that have to
 * be kept in agreement.
 *
 * There is only ever a width and a height; the ratios are shortcuts to a pair of
 * them, so picking one writes the boxes and typing in the boxes selects Custom.
 * Nothing here can hold a size the other half disagrees with, because there is
 * only one size.
 */
import type { ImageComposer } from '../store'

/**
 * 8, the VAE's downscale — the finest grid a pixel size can actually land on.
 *
 * It was 16 (VAE 8x, then patch 2 in the DiT), which is the grid the DiT wants but
 * not the one it needs: `SingleStreamDiT.forward` pads the latent up to the patch
 * size and crops the result back to the unpadded shape, so an odd latent dimension
 * costs one row of patches and nothing else. 16 was refusing every second size the
 * model can render. What has to stay true is only that the box never claims a size
 * the model will not render, which 8 still guarantees.
 */
export const SNAP = 8

const snapClamp = (v: number | string, d: number): number => {
  const n = parseInt(String(v), 10)
  return Number.isFinite(n) ? Math.max(64, Math.min(2048, Math.floor(n / SNAP) * SNAP)) : d
}
const snap8 = (v: number) => Math.max(8, Math.round(v / 8) * 8)

/**
 * The seven buckets stay exactly what they were and get *multiplied*, rather than
 * being recomputed from the ratio at a new edge length.
 *
 * Krea 2 inherits Qwen-Image's trained buckets, and 1152x896 is one of them where
 * the honest arithmetic for 4:3 at a 1024 short edge is 1365x1024 — a size nothing
 * was trained on. Scaling a bucket keeps the shape the model knows and changes
 * only how much of it there is; deriving one from the ratio would quietly leave
 * the distribution at every scale including 1x.
 */
export const BUCKETS: [string, string][] = [
  ['1024x1024', '1:1'],
  ['1152x896', '4:3'],
  ['1216x832', '3:2'],
  ['1344x768', '16:9'],
  // 3:4 is here because the swap button put it here: every other landscape preset
  // had a portrait counterpart to flip into and 4:3 did not, so the one ratio the
  // page opens on was the one whose flip landed on Custom.
  ['896x1152', '3:4'],
  ['832x1216', '2:3'],
  ['768x1344', '9:16'],
]

export const SIZE_SCALES: [string, string][] = [['1', '1K'], ['1.5', '1.5K'], ['2', '2K']]

export function readSize(img: ImageComposer): [number, number] {
  // Custom is literal. A number you typed is not a bucket to be multiplied —
  // scaling it would mean the box said 1153 and the model rendered 2306.
  if (img.aspect === 'custom') return [snapClamp(img.width, 1024), snapClamp(img.height, 1024)]
  const [w, h] = img.aspect.split('x').map(Number)
  const k = Number(img.scale) || 1
  return [snap8((w ?? 1024) * k), snap8((h ?? 1024) * k)]
}

/** Reduced, so Custom can say whether 992×1488 is still the 2:3 you meant. Only
 *  when it reduces to something a person actually says, though: 992×1024 is
 *  honestly 31:32, and printing that is noise dressed as precision. */
export function ratio(w: number, h: number): string {
  const g = (a: number, b: number): number => (b ? g(b, a % b) : a)
  const d = g(w, h) || 1
  const a = w / d
  const b = h / d
  return a <= 21 && b <= 21 ? `${a}:${b}` : ''
}

/** The button's own label. The option's text, not `ratio(w,h)`: these are trained
 *  buckets, and a bucket is not its name — 1152x896 reduces to 9:7 and 1344x768 to
 *  7:4. Both are called 4:3 and 16:9 everywhere the models are documented, and
 *  printing the honest fraction would rename two presets nobody would recognise.
 *  Custom is the only case where the arithmetic is the best name available. */
export function sizeLabel(img: ImageComposer): string {
  const [w, h] = readSize(img)
  const name = img.aspect === 'custom'
    ? (ratio(w, h) || 'Custom')
    : (BUCKETS.find(([v]) => v === img.aspect)?.[1] ?? 'Custom')
  return `${name} · ${w}×${h}`
}

/** Applying a bucket writes the boxes, because the boxes are the parameter. */
export function withAspect(img: ImageComposer, aspect: string): Partial<ImageComposer> {
  const next = { ...img, aspect }
  const [w, h] = readSize(next)
  return { aspect, width: w, height: h }
}

export function withScale(img: ImageComposer, scale: string): Partial<ImageComposer> {
  const next = { ...img, scale }
  const [w, h] = readSize(next)
  return { scale, width: w, height: h }
}

/** Snapped on the way out rather than per keystroke, which would fight you at
 *  "10" on the way to "1000". */
export function commitBoxes(img: ImageComposer): Partial<ImageComposer> {
  const [w, h] = readSize({ ...img, aspect: 'custom' })
  return { aspect: 'custom', width: w, height: h }
}

/**
 * The swap belongs to the same control, which is what decides where it lands.
 *
 * **The bucket transposes, not the pixels.** Transposing what `readSize` returns
 * looks for `1152x2016` among options that are all 1x, finds nothing, and drops a
 * perfectly ordinary 9:16 at 1.5K into Custom — so the swap silently cost you the
 * scale *and* the preset's name. The bucket's own transpose is always on the menu,
 * because the seven exist as transposed pairs.
 */
export function swapSize(img: ImageComposer): Partial<ImageComposer> {
  if (img.aspect !== 'custom') {
    const [bw, bh] = img.aspect.split('x')
    const flipped = `${bh}x${bw}`
    if (BUCKETS.some(([v]) => v === flipped)) return withAspect(img, flipped)
  }
  const [w, h] = readSize(img)
  const preset = BUCKETS.some(([v]) => v === `${h}x${w}`)
  return preset
    ? withAspect(img, `${h}x${w}`)
    : commitBoxes({ ...img, aspect: 'custom', width: h, height: w })
}

/** Which trained bucket a typed size is closest to in shape, so leaving Custom
 *  lands somewhere the model knows rather than on whichever option happened to be
 *  selected before you started typing. */
export function nearestBucket(w: number, h: number): string {
  const r = w / h
  return BUCKETS
    .map(([v]) => {
      const [a, b] = v.split('x').map(Number)
      return { v, d: Math.abs((a ?? 1) / (b ?? 1) - r) }
    })
    .sort((x, y) => x.d - y.d)[0]?.v ?? '1024x1024'
}
