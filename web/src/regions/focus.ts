/**
 * One bit that carries the caret across a render.
 *
 * The iteration loop is: type in a region's prompt, ⌘Enter, look at the result, type
 * again. But a render remounts the region layer — new job, new files, a fresh `.shot`
 * figure — so the field you were typing in is gone and rebuilt by the time the picture
 * lands, and focus is lost every single cycle. That is the whole friction the loop was
 * meant to remove, reappearing once per render.
 *
 * So when you generate *from* a region field, `ask()` sets this flag; when the card
 * remounts on the new render, it `take()`s the flag and puts the caret back. A
 * module-level boolean rather than store state on purpose: it must not cause a render,
 * it is read exactly once, and it is consumed the moment it is read so a later
 * unrelated mount does not steal focus. It is only ever set by a keystroke that already
 * had focus in the field, so returning focus there is resuming, not stealing it.
 */
let resume = false

export const askResumeFocus = (): void => { resume = true }

export const takeResumeFocus = (): boolean => {
  const v = resume
  resume = false
  return v
}
