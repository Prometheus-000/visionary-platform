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
