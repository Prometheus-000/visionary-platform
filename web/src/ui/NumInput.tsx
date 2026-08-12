/**
 * ↑ / ↓ on a number, ⌘ (or Ctrl) for the coarse step.
 *
 * 1 and 8 are the defaults because Width and Height are the boxes this was asked
 * for, and 8 is the VAE's grid — ⌘↑ there lands on the next size the model can
 * actually render rather than one it will floor. A field whose useful range is
 * 1.0 to 1.4 passes its own `step`: stepping a shift of 1.15 by 8 leaves behind
 * every value the model accepts, which is a shortcut that cannot be right once.
 * The coarse step is 8× the fine one unless `bigStep` says otherwise, so the two
 * stay in proportion wherever the fine one is overridden.
 *
 * Nothing is snapped or clamped here. The size boxes already snap on the way out
 * — typing 1153 shows 1153 until you leave the field — and an arrow is a faster
 * way to *type* a number, not a second way to commit one.
 */
import { useCallback } from 'react'

const dec = (v: number | string) => (String(v).split('.')[1] ?? '').length

export function step(
  value: string,
  dir: 1 | -1,
  coarse: boolean,
  { fine = 1, bigStep, base }: { fine?: number; bigStep?: number; base?: number } = {},
): string | null {
  const amount = coarse ? (bigStep ?? fine * 8) : fine
  // The placeholder's number when the box is empty, which is what makes ↑ on an
  // empty Steps mean "one more than the model would have used" rather than "1".
  // A seed reading "random" has no such number and is left alone — walking a
  // seed you have not drawn yet is not a thing the box can do.
  const from = value !== '' ? parseFloat(value) : base
  if (from === undefined || !Number.isFinite(from)) return null
  // Rounded to the decimals the value and the step carry between them: 1.15 +
  // 0.1 is 1.2500000000000002 in binary floating point, and the box would print
  // every digit of it.
  const next = Math.max(0, Number((from + dir * amount).toFixed(Math.max(dec(amount), dec(from)))))
  return String(next)
}

/**
 * A numeric box in the console.
 *
 * Every one of them takes the arrows, which in the vanilla page was a delegated
 * listener on three sections — because region rows are built long after load,
 * and a handler attached only at startup is one every row added later silently
 * goes without. In React the component *is* the delegation.
 *
 * Enter blurs and then generates, because you have just typed a seed or a step
 * count and reaching for the mouse to commit it is the wrong ending. The blur is
 * first so a width still on its way to being snapped is snapped before the run
 * reads it.
 */
export function NumInput({
  value,
  onValue,
  fine,
  bigStep,
  base,
  onEnter,
  onCommit,
  ...rest
}: {
  value: string
  onValue: (v: string) => void
  fine?: number
  bigStep?: number
  /** What an empty box counts from — the checkpoint's own number. */
  base?: number
  onEnter?: () => void
  /** Blur and Enter, for the boxes that snap on the way out. */
  onCommit?: () => void
} & Omit<React.InputHTMLAttributes<HTMLInputElement>, 'value' | 'onChange' | 'step'>) {
  const onKeyDown = useCallback(
    (e: React.KeyboardEvent<HTMLInputElement>) => {
      if (e.key === 'Enter' && !e.nativeEvent.isComposing) {
        e.preventDefault()
        onCommit?.()
        e.currentTarget.blur()
        onEnter?.()
        return
      }
      if (e.key !== 'ArrowUp' && e.key !== 'ArrowDown') return
      if (e.altKey || e.shiftKey) return
      const next = step(value, e.key === 'ArrowUp' ? 1 : -1, e.metaKey || e.ctrlKey,
                        { fine, bigStep, base })
      if (next === null) return
      e.preventDefault()
      onValue(next)
    },
    [value, onValue, fine, bigStep, base, onEnter, onCommit],
  )

  return (
    <input {...rest} value={value} inputMode={rest.inputMode ?? 'decimal'}
           onChange={(e) => onValue(e.target.value)}
           onBlur={(e) => { onCommit?.(); rest.onBlur?.(e) }}
           onKeyDown={onKeyDown} />
  )
}
