/**
 * The field fits its content, up to three lines; past that it scrolls.
 *
 * **This retires `CONSOLE_BUDGET = 0.3`, on the owner's ruling, and the ruling
 * is worth quoting because the number will look like a lost invariant to
 * whoever misses it:** "That number has caused me more trouble than it's
 * worth. It was never meant to be exact. The point was that the canvas should
 * always be dominant — the second it's not, the platform becomes utilitarian.
 * The prompt box should grow and contract based on content. There will never
 * be a reason for it to come close to 50%, and if that reason surfaces one
 * day, that's simply a UI layout problem that needs to be designed for."
 *
 * So the budget arithmetic is gone, not relaxed. The old cap was *derived* —
 * whatever 30% of this viewport had left after the rest of the console — which
 * meant it landed on an arbitrary pixel at almost every window size, needed a
 * whole-lines quantiser to keep the caret out of half-rendered rows, needed a
 * floor for the viewports where the arithmetic left less than a writable box,
 * needed a ResizeObserver because the answer moved when a rail wrapped, and
 * still broke the field once it hit three lines when one of those pieces
 * measured the wrong element. Every one of those subsystems existed to defend
 * a number that was never the invariant. A cap stated in *lines* is exact by
 * construction: `chrome + 3 × line` is on a line boundary always, at every
 * viewport, before any font loads.
 *
 * What actually holds the canvas dominant is the console's own overflow cap —
 * CSS, not this file — and `tools/ui-checks/probe_console.py` now watches
 * dominance itself: it reports the console's share and fails past 50%, loudly,
 * as the design problem the ruling says it would be. Never re-add a clamp here
 * to make that probe quiet — see the memory rule "a broken budget is
 * information": when a limit and the UI disagree, one of them is wrong, and
 * the answer is a diagnosis, not a squeeze.
 */
export const FIELD_LINES = 3

/**
 * When line metrics are unreadable — `line-height: normal` parses to NaN, and
 * a font that has not loaded can report zero — the cap is better than a box of
 * no height. Three 21px lines on 8px of chrome, near enough.
 */
const FALLBACK_CAP = 72

/** `chrome + FIELD_LINES × line`, read off the element's own computed style
 *  because the quantum differs between the prompt and a shot row. */
function cap(el: HTMLTextAreaElement): number {
  const cs = getComputedStyle(el)
  const line = parseFloat(cs.lineHeight)
  // `box-sizing: border-box`, so the height being set includes both.
  const chrome = parseFloat(cs.paddingTop) + parseFloat(cs.paddingBottom)
    + parseFloat(cs.borderTopWidth) + parseFloat(cs.borderBottomWidth)
  if (!Number.isFinite(line) || line <= 0 || !Number.isFinite(chrome)) return FALLBACK_CAP
  return chrome + FIELD_LINES * line
}

/**
 * Grow to fit, up to three lines; past them, scroll.
 *
 * The `height = 'auto'` first is not a tidy-up — `scrollHeight` on an element
 * with an explicit height reports that height, so without it the field can
 * only ever grow. Content below the cap lands on a whole line because
 * `scrollHeight` *is* `chrome + n × line`; content above it gets the cap,
 * which is on a line boundary by construction — the caret can never sit in a
 * half-rendered row, which is the bug the old quantiser existed to hold off.
 */
export function autoGrow(fieldEl: HTMLTextAreaElement | null): void {
  if (!fieldEl) return
  fieldEl.style.height = 'auto'
  fieldEl.style.height = `${Math.min(fieldEl.scrollHeight, cap(fieldEl))}px`
}

/**
 * The same rule per shot row — each fits its own content, up to three lines.
 *
 * Rows used to *divide* one budget-derived allowance, so a fourth shot shrank
 * the other three toward a 30px floor you could not read the current line of.
 * Content-sized rows make a tall scene an honest cost instead of a mutual
 * squeeze; the console's own overflow cap is what keeps the canvas dominant
 * when somebody writes six three-line shots, and if that state ever matters
 * it is a layout to design, not a reason to shrink a box someone is typing in.
 */
export function growRows(box: HTMLElement | null): void {
  if (!box) return
  const areas = [...box.querySelectorAll('textarea')]
  if (!areas.length) return
  // `auto` first for the reason `autoGrow` does it: measuring without this
  // lets a row grow and never shrink.
  for (const a of areas) a.style.height = 'auto'
  for (const a of areas) a.style.height = `${Math.min(a.scrollHeight, cap(a))}px`
}

/**
 * Grow whichever surface is currently the field.
 *
 * There are three and only one is ever on screen: the negative box when you
 * have switched into it, the shot rows on the video side, and the prompt on
 * the image side. This lived as a `querySelector` chain inside `Console`'s
 * ResizeObserver and had to be repeated by every caller that could change a
 * height — which is how a fourth surface gets added and grows only when the
 * window is resized.
 */
export function growField(consoleEl: HTMLElement | null): void {
  if (!consoleEl) return
  const neg = consoleEl.querySelector<HTMLTextAreaElement>('.field.on-neg #neg')
  if (neg) { autoGrow(neg); return }
  const rows = consoleEl.querySelector<HTMLElement>('.tline:not(.hide)')
  if (rows) { growRows(rows); return }
  autoGrow(consoleEl.querySelector<HTMLTextAreaElement>('#prompt'))
}
