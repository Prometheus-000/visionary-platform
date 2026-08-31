/**
 * Fit whatever is on the canvas into the height the console left it.
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

/**
 * What is left of the canvas once its padding and everything under it is gone.
 *
 * Shared, because a clip and a still are fitted into the same box by the same
 * arithmetic and the second copy of it is where the two drift apart. That is
 * not hypothetical here: the still had this and the clip had nothing at all.
 */
function room(
  canvas: HTMLElement,
  caption: HTMLElement | null,
  nav?: HTMLElement | null,
): number {
  const cs = getComputedStyle(canvas)
  return (
    canvas.clientHeight -
    parseFloat(cs.paddingTop) -
    parseFloat(cs.paddingBottom) -
    outer(caption) -
    outer(nav)
  )
}

/**
 * An element's height *including its margins*, or zero.
 *
 * **The flat `- 12` this replaces was the caption's margin, guessed at half.**
 * `#gen-meta` and `#vid-meta` both carry `margin: 12px 2px`, so the space under
 * a render is 24px of margin and `offsetHeight` reports none of it — the fit
 * came out 12px short and the canvas scrolled by exactly that. Invisible on the
 * still, where `ACTS_H` is generous enough to absorb it; the clip has no acts
 * strip, so there it was the whole remaining overflow.
 *
 * The file's own rule, applied to the one subtrahend that was still a constant:
 * *every subtrahend is measured rather than guessed* — written after a fudge
 * factor came out 42px short, and left in place next to a 12.
 */
function outer(el: HTMLElement | null | undefined): number {
  if (!el) return 0
  const cs = getComputedStyle(el)
  return el.offsetHeight + parseFloat(cs.marginTop) + parseFloat(cs.marginBottom)
}

export function layoutShots(
  film: HTMLElement | null,
  canvas: HTMLElement | null,
  caption: HTMLElement | null,
  n: number,
  nav?: HTMLElement | null,
): void {
  if (!film || !canvas || !n) return
  film.style.setProperty('--shot-h', `${room(canvas, caption, nav) - ACTS_H}px`)
}

/**
 * The same fit, for a clip.
 *
 * **The video canvas had no fit at all, and a 720p take did not fit in it.**
 * `#vid-out video` was `width:100%;max-width:1180px` with no ceiling on the
 * height, so a 16:9 clip was 665px tall wherever the console had left 360 — the
 * canvas scrolled, the native controls were cut off below the fold, and the
 * metadata line under it was somewhere off-screen. Every rule this app has about
 * the canvas — *the largest thing on screen, always*, *the result on screen is
 * the canvas* — was true of stills and false of the thing the video side exists
 * to show.
 *
 * It stayed invisible because it could not be seen: `preview_ui.py` served an
 * SVG for every `.mp4`, so `<video>` reported no intrinsic size and laid out at
 * the height of its own controls. Both halves are fixed — see `CLIP_720P`.
 *
 * No `ACTS_H`: a clip has no per-still actions strip under it. Its own zoom
 * button floats over the top-right corner and costs the layout nothing.
 */
export function layoutClip(
  box: HTMLElement | null,
  canvas: HTMLElement | null,
  caption: HTMLElement | null,
): void {
  if (!box || !canvas) return
  box.style.setProperty('--clip-h', `${room(canvas, caption)}px`)
}
