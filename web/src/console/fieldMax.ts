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
 * @param consoleEl the `.console` element
 * @param fieldEl   the field currently on show — the negative box when that is
 *                  the one visible, which is what `liveField()` decided
 */
export function fieldMax(consoleEl: HTMLElement | null, fieldEl: HTMLElement | null): number {
  if (!consoleEl || !fieldEl) return FIELD_CEIL
  const other = consoleEl.getBoundingClientRect().height - fieldEl.getBoundingClientRect().height
  return Math.max(FIELD_FLOOR, Math.min(FIELD_CEIL, window.innerHeight * CONSOLE_BUDGET - other))
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
  fieldEl.style.height = `${Math.min(fieldEl.scrollHeight, fieldMax(consoleEl, fieldEl))}px`
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
  const share = Math.max(ROW_FLOOR, fieldMax(consoleEl, box) / areas.length)
  for (const a of areas) a.style.height = `${Math.min(a.scrollHeight, share)}px`
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
