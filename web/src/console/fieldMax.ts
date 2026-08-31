/**
 * The prompt field gets whatever the console budget has left, down to a floor.
 *
 * Ported verbatim from UI_HTML, constants included, because
 * `tools/ui-checks/baseline/console.json` pins the result at three viewports
 * and seven states — this is the first thing in the port that can be held to a
 * number rather than to a screenshot.
 *
 * Why it is not a constant cap: the console has a 30% budget and everything
 * else in it is fixed or conditional — the strip is one row, the rail appears
 * with the first pill, and the boxes cost it nothing at all. The prompt is the
 * only part that grows without asking, and measuring showed it was also the
 * part that broke the budget alone: at a flat 168px cap the worst case was
 * 39.8% of a 1440x900 window, 136 of which was the field.
 *
 * The floor is the half that is easy to delete by accident. Below two lines
 * the box stops being somewhere you can write, so `FIELD_FLOOR` is allowed to
 * win and the console is allowed over budget — at 1440x900 it lands at 31.6%
 * and that is correct. A check that asserts `<= 30%` reports this as a bug on
 * the shortest viewport anyone uses, which is why the baseline pins the
 * formula instead.
 */
export const CONSOLE_BUDGET = 0.3
export const FIELD_FLOOR = 52
export const FIELD_CEIL = 168

/**
 * The allowance, given how tall the field part currently measures.
 *
 * Split out from `fieldMax` because the video side's field is *several*
 * elements — see `growRows`, where handing this a container was the bug.
 */
function allowance(consoleEl: HTMLElement | null, fieldHeight: number): number {
  if (!consoleEl) return FIELD_CEIL
  const other = consoleEl.getBoundingClientRect().height - fieldHeight
  return Math.max(FIELD_FLOOR, Math.min(FIELD_CEIL, window.innerHeight * CONSOLE_BUDGET - other))
}

/**
 * @param consoleEl the `.console` element
 * @param fieldEl   the field currently on show — the negative box when that is
 *                  the one visible, which is what `liveField()` decided
 */
export function fieldMax(consoleEl: HTMLElement | null, fieldEl: HTMLElement | null): number {
  if (!consoleEl || !fieldEl) return FIELD_CEIL
  return allowance(consoleEl, fieldEl.getBoundingClientRect().height)
}

/**
 * The cap, rounded **down to a whole number of lines**.
 *
 * **A box that is 2.095 lines tall is the "the cursor goes whacky" bug**, and it
 * is not cosmetic. Every constant here is a pixel count — 52, 168, 30 — and a
 * line is 21px on 4px of padding, so none of them lands on a line boundary:
 * `FIELD_FLOOR` measured 2.095 lines, which means the box permanently shows a
 * 2px sliver of the line below. Once the text is long enough to scroll, the
 * browser scrolls by whatever it takes to keep the caret in view — 14.5px,
 * measured — and the caret then sits inside a half-rendered line with the row
 * above it sliced through the middle. You are typing into a strip of glyphs cut
 * in half, which is what "3 lines and it goes whacky" is.
 *
 * It cannot be fixed by choosing rounder constants. The cap is derived — it is
 * whatever 30% of *this* viewport has left after the rest of the console — so it
 * lands on an arbitrary pixel at almost every window size. The height has to be
 * quantised where it is applied, which is here.
 *
 * Rounded *down*, never up: up is a box one line taller than the budget allows,
 * and the budget is the thing all of this exists to hold. One line is the floor
 * — a box with no line in it is not a box you can write in — and it is allowed
 * to exceed the cap for the same reason `FIELD_FLOOR` is.
 */
function wholeLines(el: HTMLTextAreaElement, cap: number): number {
  const cs = getComputedStyle(el)
  const line = parseFloat(cs.lineHeight)
  // `box-sizing: border-box`, so the height being set includes both.
  const chrome = parseFloat(cs.paddingTop) + parseFloat(cs.paddingBottom)
    + parseFloat(cs.borderTopWidth) + parseFloat(cs.borderBottomWidth)
  // `line-height: normal` parses to NaN, and a font that has not loaded can
  // report zero. Either way the cap is better than a box of no height.
  if (!Number.isFinite(line) || line <= 0 || !Number.isFinite(chrome)) return cap
  return chrome + Math.max(1, Math.floor((cap - chrome) / line)) * line
}

/**
 * Grow to fit, up to the cap; past it, scroll.
 *
 * The `height = 'auto'` first is not a tidy-up — `scrollHeight` on an element
 * with an explicit height reports that height, so without it the field can
 * only ever grow.
 */
export function autoGrow(fieldEl: HTMLTextAreaElement | null, consoleEl: HTMLElement | null): void {
  if (!fieldEl) return
  fieldEl.style.height = 'auto'
  const cap = wholeLines(fieldEl, fieldMax(consoleEl, fieldEl))
  fieldEl.style.height = `${Math.min(fieldEl.scrollHeight, cap)}px`
}

/**
 * The same allowance, divided between shot rows.
 *
 * **Rows divide the field's allowance rather than adding to it.** One prompt at
 * two lines and two shots at one line each are the same height, so a four-shot
 * scene costs exactly what a long prompt costs and the 30% budget holds without
 * a second arithmetic. With one row it reduces to `autoGrow` exactly — same
 * cap, same measurement — which is what keeps the single-shot case byte-for-byte
 * the prompt box it replaces.
 *
 * `ROW_FLOOR` is one line plus the padding `.field textarea` already spends. A
 * share below that is a row you cannot read the current line of, and at that
 * point the box scrolls rather than shrinking further — the same trade
 * `FIELD_FLOOR` makes for the single field.
 */
export const ROW_FLOOR = 30

export function growRows(box: HTMLElement | null, consoleEl: HTMLElement | null): void {
  if (!box) return
  const areas = [...box.querySelectorAll('textarea')]
  if (!areas.length) return
  // `auto` first for the same reason `autoGrow` does it: `scrollHeight` on an
  // element with an explicit height reports that height, so measuring without
  // this lets a row grow and never shrink.
  for (const a of areas) a.style.height = 'auto'
  // **The rows, not the box holding them.** This measured `fieldMax(console,
  // box)` — and `.tline` is the strip and the ruler *as well as* the row, so
  // everything `Timeline` draws was subtracted out of `other` and then handed
  // back to the textarea as if it were the textarea's own. The allowance came
  // out one Timeline too generous at every viewport, and the console ran 36% of
  // a window it is budgeted 30% of. `probe_console.py` flagged it on every video
  // row as `FIELD vs fieldMax()`; it is the one state the formula was never
  // right for, because it is the one field that is not a single element.
  const rows = areas.reduce((n, a) => n + a.getBoundingClientRect().height, 0)
  const share = Math.max(ROW_FLOOR, allowance(consoleEl, rows) / areas.length)
  for (const a of areas) {
    a.style.height = `${Math.min(a.scrollHeight, wholeLines(a, share))}px`
  }
}

/**
 * Grow whichever surface is currently the field.
 *
 * There are three and only one is ever on screen: the negative box when you
 * have switched into it, the shot rows on the video side, and the prompt on the
 * image side. This lived as a `querySelector` chain inside `Console`'s
 * ResizeObserver and had to be repeated by every caller that could change a
 * height — which is how a fourth surface gets added and grows only when the
 * window is resized.
 */
export function growField(consoleEl: HTMLElement | null): void {
  if (!consoleEl) return
  const neg = consoleEl.querySelector<HTMLTextAreaElement>('.field.on-neg #neg')
  if (neg) { autoGrow(neg, consoleEl); return }
  const rows = consoleEl.querySelector<HTMLElement>('.tline:not(.hide)')
  if (rows) { growRows(rows, consoleEl); return }
  autoGrow(consoleEl.querySelector<HTMLTextAreaElement>('#prompt'), consoleEl)
}
