/**
 * The character reference sheet — template, layout and the PNG itself.
 *
 * **The format is MiniMax's own.** Their `3d-animation-short-generator` skill
 * specifies a 16:9 production reference sheet with readable labels drawn on the
 * image — name, role, a main ¾ view, front/side/back views, expressions,
 * costume and prop details, and a short visual-ID note — *"so downstream
 * generation can bind the correct person and props."* Labels on the picture are
 * a feature, not decoration: H3's conditioner reads text, and the sheet's whole
 * advantage over loose photographs is that one `<Picture N>` slot carries the
 * views *and* the words binding them.
 *
 * **One layout function feeds both the preview and the export.** The builder's
 * canvas renders exactly the PNG that will be saved — a preview with its own
 * arrangement is a preview that can disagree with the file, which is the
 * `/api/compile` rule one surface over.
 *
 * **Filled panels only, and few big panels beat many small ones.** The
 * community guide's one quality warning: "large, sharp subjects and clearly
 * separated sections give H3 stronger visual information than a crowded page of
 * tiny panels." So empty slots are simply not drawn, and the arrangement
 * reflows to spend the whole sheet on what is there — an export with dashed
 * placeholder rectangles would be a sheet teaching the model about dashed
 * rectangles.
 */

export const SHEET_W = 1920
export const SHEET_H = 1080

/** The template's six slots, in the order the skill lists them. `main` is the
 *  hero and takes the left column whenever it is filled; a turnaround
 *  (front/side/back) prefers full-height columns because figures are portrait. */
export const SHEET_SLOTS = [
  { id: 'main', label: '¾ VIEW' },
  { id: 'front', label: 'FRONT' },
  { id: 'side', label: 'SIDE' },
  { id: 'back', label: 'BACK' },
  { id: 'expr', label: 'EXPRESSIONS' },
  { id: 'props', label: 'COSTUME & PROPS' },
] as const
export type SlotId = (typeof SHEET_SLOTS)[number]['id']

export type SheetSpec = {
  name: string
  role: string
  /** The visual-ID note the skill asks for: age range, body type, hairstyle,
   *  outfit colours, signature props, do-not-change traits. */
  note: string
  images: Partial<Record<SlotId, HTMLImageElement>>
}

const M = 24            // outer margin
const GAP = 16          // between panels
const HEADER = 104      // the name band
const LABEL = 36        // each panel's caption band

type Rect = { x: number; y: number; w: number; h: number }

/**
 * Panel rectangles for the filled slots, keyed by slot id.
 *
 * The arrangements are chosen for figures rather than derived from a packing
 * formula: three companions become full-height *columns* because that is what
 * a turnaround is, and a formula that optimised coverage would happily make
 * front/side/back three letterboxed strips.
 */
export function layout(filled: SlotId[]): Record<string, Rect> {
  const out: Record<string, Rect> = {}
  const bodyY = M + HEADER + GAP
  const bodyH = SHEET_H - bodyY - M
  const bodyW = SHEET_W - 2 * M
  if (!filled.length) return out

  const hero = filled[0]!
  const rest = filled.slice(1)

  const grid = (ids: SlotId[], x: number, w: number, cols: number) => {
    const rows = Math.ceil(ids.length / cols)
    const rh = (bodyH - GAP * (rows - 1)) / rows
    ids.forEach((id, i) => {
      const last = ids.length % cols
      const r = Math.floor(i / cols)
      const c = i % cols
      // The final row stretches to fill the width — two panels under three
      // columns as two wider panels, not two panels and a hole.
      const inRow = r === rows - 1 && last ? last : cols
      const w2 = (w - GAP * (inRow - 1)) / inRow
      out[id] = { x: x + c * (w2 + GAP), y: bodyY + r * (rh + GAP), w: w2, h: rh }
    })
  }

  if (!rest.length) {
    out[hero] = { x: M, y: bodyY, w: bodyW, h: bodyH }
  } else if (rest.length <= 2) {
    const hw = Math.round(bodyW * 0.56)
    out[hero] = { x: M, y: bodyY, w: hw, h: bodyH }
    grid(rest, M + hw + GAP, bodyW - hw - GAP, 1)
  } else if (rest.length === 3) {
    const hw = Math.round(bodyW * 0.4)
    out[hero] = { x: M, y: bodyY, w: hw, h: bodyH }
    grid(rest, M + hw + GAP, bodyW - hw - GAP, 3)
  } else {
    const hw = Math.round(bodyW * 0.38)
    out[hero] = { x: M, y: bodyY, w: hw, h: bodyH }
    grid(rest, M + hw + GAP, bodyW - hw - GAP, rest.length === 4 ? 2 : 3)
  }
  return out
}

/**
 * Draw the whole sheet onto `ctx`. Synchronous — the caller owns image loading,
 * because the builder redraws on every change and images load once.
 *
 * Light ground, dark ink. Production sheets are read, not admired: the
 * neutral near-white keeps every costume colour honest, where a dark ground
 * would recolour everything against it.
 */
export function drawSheet(ctx: CanvasRenderingContext2D, spec: SheetSpec): void {
  ctx.fillStyle = '#fafafa'
  ctx.fillRect(0, 0, SHEET_W, SHEET_H)

  // ── header ────────────────────────────────────────────────────────────────
  const name = spec.name.trim() || 'Unnamed'
  ctx.fillStyle = '#111'
  ctx.font = '600 54px -apple-system, "Helvetica Neue", Arial, sans-serif'
  ctx.textBaseline = 'alphabetic'
  ctx.fillText(name.toUpperCase(), M, M + 58)
  let cursor = M + ctx.measureText(name.toUpperCase()).width

  if (spec.role.trim()) {
    // The role as a chip, the way the skill labels it — protagonist, thief,
    // grandma — a category beside the name, not a second name.
    ctx.font = '500 26px -apple-system, "Helvetica Neue", Arial, sans-serif'
    const role = spec.role.trim().toUpperCase()
    const w = ctx.measureText(role).width
    const rx = cursor + 24
    ctx.fillStyle = '#e8e8e8'
    ctx.beginPath()
    ctx.roundRect(rx, M + 24, w + 28, 40, 20)
    ctx.fill()
    ctx.fillStyle = '#444'
    ctx.fillText(role, rx + 14, M + 52)
    cursor = rx + w + 28
  }

  // The visual-ID note, right-aligned, wrapped to at most two lines. It is
  // the do-not-change list, so it is prose on the sheet rather than metadata
  // beside it — the model sees the sheet, not our store.
  if (spec.note.trim()) {
    ctx.font = '400 24px -apple-system, "Helvetica Neue", Arial, sans-serif'
    ctx.fillStyle = '#555'
    const maxW = SHEET_W - M - (cursor + 48)
    const words = spec.note.trim().split(/\s+/)
    const lines: string[] = []
    let line = ''
    for (const w of words) {
      const t = line ? `${line} ${w}` : w
      if (ctx.measureText(t).width > maxW && line) {
        lines.push(line)
        line = w
        if (lines.length === 2) break
      } else line = t
    }
    if (lines.length < 2 && line) lines.push(line)
    ctx.textAlign = 'right'
    lines.slice(0, 2).forEach((l, i) => {
      ctx.fillText(l, SHEET_W - M, M + 34 + i * 32)
    })
    ctx.textAlign = 'left'
  }

  ctx.strokeStyle = '#ddd'
  ctx.lineWidth = 2
  ctx.beginPath()
  ctx.moveTo(M, M + HEADER)
  ctx.lineTo(SHEET_W - M, M + HEADER)
  ctx.stroke()

  // ── panels ────────────────────────────────────────────────────────────────
  const filled = SHEET_SLOTS.filter((s) => spec.images[s.id]).map((s) => s.id)
  const rects = layout(filled)
  for (const slot of SHEET_SLOTS) {
    const rect = rects[slot.id]
    const img = spec.images[slot.id]
    if (!rect || !img) continue
    ctx.fillStyle = '#fff'
    ctx.fillRect(rect.x, rect.y, rect.w, rect.h)
    ctx.strokeStyle = '#d8d8d8'
    ctx.lineWidth = 1.5
    ctx.strokeRect(rect.x + 0.75, rect.y + 0.75, rect.w - 1.5, rect.h - 1.5)

    // Contain, never cover — the same rule the gallery's thumbnails keep, for
    // the same reason: cropping throws away the thing the panel exists to
    // show, and a reference sheet cropping a costume is a sheet lying about it.
    const ih = rect.h - LABEL
    const s = Math.min((rect.w - 12) / img.naturalWidth, (ih - 12) / img.naturalHeight)
    const dw = img.naturalWidth * s
    const dh = img.naturalHeight * s
    ctx.drawImage(img, rect.x + (rect.w - dw) / 2, rect.y + (ih - dh) / 2, dw, dh)

    // The caption band. Small caps, quiet, and *on* the image file — the
    // whole point of a sheet over loose photographs is that the words travel
    // with the pixels.
    ctx.fillStyle = '#f1f1f1'
    ctx.fillRect(rect.x, rect.y + rect.h - LABEL, rect.w, LABEL)
    ctx.fillStyle = '#666'
    ctx.font = '600 21px -apple-system, "Helvetica Neue", Arial, sans-serif'
    ctx.textAlign = 'center'
    ctx.fillText(slot.label, rect.x + rect.w / 2, rect.y + rect.h - 11)
    ctx.textAlign = 'left'
  }
}
