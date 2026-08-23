import { useEffect, useState } from 'react'

/**
 * Is a pointer currently held down?
 *
 * **A control must not act on a click whose press it never received.** That is
 * the same reasoning as this codebase's "mousedown, not click" rule, inverted:
 * there the element is *replaced* before the click arrives, here it *appears*
 * before the click arrives, and both end with a click landing on something that
 * was not what anybody pressed.
 *
 * The fault it was written for: the mention menu picks on mousedown — it has to,
 * because the textarea has focus and a click would blur it first and close the
 * menu on the way to firing — so a cast card mounts at the cursor mid-gesture,
 * and the mouseup of that same press landed on whichever slot was now underneath.
 * Adding a character opened a file dialog for Motion, and the card nobody had
 * seen yet was behind it.
 *
 * A flag rather than a per-control guard, because the exposure is not the slot's:
 * the card's ✕ is under the same cursor, and that one deletes the thing you just
 * made.
 */
let down = false

if (typeof document !== 'undefined') {
  // Capture, so this is true before any handler that might mount something asks.
  document.addEventListener('pointerdown', () => { down = true }, true)
  document.addEventListener('pointerup', () => { down = false }, true)
  document.addEventListener('pointercancel', () => { down = false }, true)
}

/**
 * False while a press that started before this mounted is still held.
 *
 * Immediately true when nothing is down — a card opened from the keyboard has no
 * gesture to wait out, and delaying it by a frame would be a card you cannot
 * type into for a frame.
 */
export function useSettled(): boolean {
  const [settled, setSettled] = useState(!down)
  useEffect(() => {
    if (!down) return
    const up = () => { setSettled(true) }
    document.addEventListener('pointerup', up, { once: true, capture: true })
    document.addEventListener('pointercancel', up, { once: true, capture: true })
    return () => {
      document.removeEventListener('pointerup', up, true)
      document.removeEventListener('pointercancel', up, true)
    }
  }, [])
  return settled
}
