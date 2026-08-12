/**
 * Fit the stills into whatever height the console left the canvas.
 *
 * Measured off the canvas, never the viewport. The console under it grows and
 * shrinks with what is open, so a `dvh` sum is wrong the moment a popover
 * opens — the stills run under the bar instead of fitting above it. A batch has
 * to fit to be compared; a single still gets the lot.
 *
 * Every subtrahend is measured rather than guessed. `clientHeight` includes the
 * canvas padding, and the caption below the grid is a real element with a real
 * height; a fixed fudge factor for the two of them was 42px short, which put
 * the caption under the console on exactly the shot you would screenshot.
 */

/** 22px of actions strip under each still, matching `.shot{padding-bottom}`.
 *  Subtracted per row rather than once, because a batch is two rows and each
 *  of them has a strip. */
const ACTS_H = 30

export function layoutShots(
  grid: HTMLElement | null,
  canvas: HTMLElement | null,
  caption: HTMLElement | null,
  n: number,
): void {
  if (!grid || !canvas || !n) return
  grid.style.gridTemplateColumns = n <= 1 ? 'minmax(0,1fr)' : 'repeat(2,minmax(0,1fr))'
  grid.style.maxWidth = n <= 1 ? '920px' : '1320px'

  const cs = getComputedStyle(canvas)
  const h =
    canvas.clientHeight -
    parseFloat(cs.paddingTop) -
    parseFloat(cs.paddingBottom) -
    (caption?.offsetHeight || 0) -
    12

  grid.style.setProperty('--shot-h', `${n <= 2 ? h - ACTS_H : h / 2 - 16 - ACTS_H}px`)
}
