/**
 * Where the caret was when `+ LoRA` was pressed, and which field had it.
 *
 * Clicking the button takes focus away, and the menu takes it again, so by the
 * time an item is chosen `selectionStart` is meaningless — it would put every
 * pick at the end of the sentence, which is the one place the syntax makes no
 * difference.
 *
 * Two fields, not one, since a region's prompt takes the same syntax at a smaller
 * scope. Tracking which was last touched is what makes "select a box, click
 * + LoRA, pick a name" put that character in that box — the shortest path this
 * feature has, and the only one that involves no typing at all.
 *
 * A module-level ref rather than store state: this changes on every keyup and
 * click in two text fields, which is pointer rate, and nothing renders from it.
 */
import type { Written } from './tokens'

type Field = HTMLTextAreaElement | HTMLInputElement

type Sink = {
  el: Field
  /** `1.3` in a region and `1` in the main prompt — see `insertLora`. */
  scope: 'prompt' | 'region'
  /** Push the new text back through React. The selection is restored after the
   *  commit, not before: these are controlled inputs, so writing `el.value`
   *  directly would be overwritten by the next render. */
  write: (value: string) => void
}

let sink: Sink | null = null
let pos = 0

/** Spread onto a prompt field. Five events, because focus alone misses a click
 *  that only moves the caret and keyup alone misses a selection made with the
 *  mouse. */
export function caretProps(scope: Sink['scope'], write: Sink['write']) {
  const remember = (e: React.SyntheticEvent<Field>) => {
    sink = { el: e.currentTarget, scope, write }
    pos = e.currentTarget.selectionStart ?? 0
  }
  return { onKeyUp: remember, onClick: remember, onSelect: remember, onFocus: remember }
}

export const caretScope = (): Sink['scope'] | null => sink?.scope ?? null
export const caretValue = (): string => sink?.el.value ?? ''
export const caretAt = (): number => pos

/** Apply a rewrite and put the caret back where the rewrite wants it. */
export function applyWrite(w: Written): void {
  const s = sink
  if (!s) return
  pos = w.caret
  s.write(w.value)
  // After the commit. React has not painted the new value yet at this point, so
  // setting the range now would range over the old text — and on a shortened
  // string the browser clamps it, which is how a pick that removed a token used
  // to leave the caret at the end of the sentence.
  requestAnimationFrame(() => {
    s.el.focus()
    s.el.setSelectionRange(w.caret, w.select)
  })
}

/** Forget the field when it goes — a region's prompt lives in the card its box
 *  opens, so it is torn down and rebuilt as the selection moves, and a stale
 *  element would keep taking writes that paint nowhere. */
export function dropCaret(el: Field): void {
  if (sink?.el === el) sink = null
}
