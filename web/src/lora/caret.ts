/**
 * Where the caret is, and which field has it.
 *
 * **It was `+ LoRA`'s, and `+ LoRA` no longer writes anything.** The button used
 * to insert `<lora:name:1>` at the caret, so it needed to know where the caret
 * had been before the menu stole focus — and two fields, because a region's
 * prompt took the same syntax at a smaller scope. Both of those are gone: a LoRA
 * is a chip now and a region has its own dropdown, so nothing picks a *place* to
 * write a LoRA any more.
 *
 * What still needs the sink is `moveClause` — ⌥←/⌥→ moving the clause under the
 * caret — and the region field registering itself so that chord works there too.
 *
 * A module-level ref rather than store state: this changes on every keyup and
 * click in two text fields, which is pointer rate, and nothing renders from it.
 */

type Field = HTMLTextAreaElement | HTMLInputElement

/** A rewritten field value with the selection to restore after it commits. */
export type Written = { value: string; caret: number; select: number }

type Sink = {
  el: Field
  /** Which field, so a chord can be scoped to one of them. */
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
