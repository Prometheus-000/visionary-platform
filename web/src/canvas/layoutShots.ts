/**
 * Fit the shown still into whatever height the console left the canvas.
 *
 * Measured off the canvas, never the viewport. The console under it grows and
 * shrinks with what is open, so a `dvh` sum is wrong the moment a popover
 * opens — the still runs under the bar instead of fitting above it.
 *
 * Every subtrahend is measured rather than guessed. `clientHeight` includes the
 * canvas padding, and the caption below is a real element with a real height; a
 * fixed fudge factor for the two of them was 42px short, which put the caption
 * under the console on exactly the shot you would screenshot.
 *
 * **A batch no longer divides this.** It used to lay four stills out as a two-up
 * contact sheet at half height each, which is what made a batch the one case the
 * region boxes could not be drawn on: there is no single surface to draw on, and
 * the boxes landed on the first cell of the four you were trying to compare. One
 * canvas at a time is the rule now, so every result gets the whole height and the
 * strip decides which one you are looking at. The nav is the other subtrahend, for
 * the same reason the caption is one.
 */

/** 22px of actions strip under the still, matching `.shot{padding-bottom}`. */
const ACTS_H = 30

export function layoutShots(
  film: HTMLElement | null,
  canvas: HTMLElement | null,
  caption: HTMLElement | null,
  n: number,
  nav?: HTMLElement | null,
): void {
  if (!film || !canvas || !n) return

  const cs = getComputedStyle(canvas)
  const h =
    canvas.clientHeight -
    parseFloat(cs.paddingTop) -
    parseFloat(cs.paddingBottom) -
    (caption?.offsetHeight || 0) -
    (nav?.offsetHeight || 0) -
    12

  film.style.setProperty('--shot-h', `${h - ACTS_H}px`)
}
