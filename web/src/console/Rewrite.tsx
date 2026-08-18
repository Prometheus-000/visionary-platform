import { failed } from '../api/client'
import { rewrite } from '../api/routes'
import { stripLoras } from '../lora/tokens'
import { useStore } from '../store'

/**
 * The third door into the prompt, beside `+ LoRA` and `Shot`.
 *
 * **In the strip, not on a row of its own.** It shipped as `#rewrite-rail` — a
 * flex row under the textarea holding one button — which is the exact price
 * `#shot-rail:empty` was written to refuse: a line per button, 34px at rest,
 * forever, out of a console capped at 30% of the viewport. Shot paid that once
 * and came back into the strip for it. This did not need to learn it twice.
 *
 * So it is shaped like its neighbours rather than like itself. `+ LoRA` writes a
 * token at the caret, `Shot` opens a vocabulary, this rewrites the sentence —
 * three doors onto the same box, and they read as three of a kind because they
 * are built as three of a kind. Same `.s`, same height, same spacing.
 *
 * **Disabled rather than absent when there is nothing to act on.** Hiding it
 * would reflow the strip on the first keystroke and move `+ LoRA` under the
 * pointer on its way to being pressed. A door that is visibly shut is also the
 * only version of this that says the feature exists before you have typed.
 */
export function Rewrite() {
  const s = useStore()
  const ops = s.state?.rewrite_ops ?? []
  const prose = stripLoras(s.prompt).trim()
  if (!ops.length) return null

  const run = async (op: string) => {
    if (s.rewriting) return
    s.setRewriting(op)
    try {
      const r = await rewrite({ prose, op })
      // `failed` is the transport; `ok` is the route's own verdict. The route
      // answers with the original text on either, so the worst case here is the
      // box unchanged rather than the box emptied.
      if (!failed(r) && r.ok && r.text) s.applyRewrite(r.text)
    } finally {
      s.setRewriting(null)
    }
  }

  return (
    <>
      {ops.map((o) => (
        <button key={o.key} className="s" type="button"
                style={{ height: 32, padding: '0 10px' }}
                disabled={!prose || !!s.rewriting}
                aria-busy={s.rewriting === o.key}
                title={o.note}
                onClick={() => void run(o.key)}>
          {s.rewriting === o.key ? `${o.label}…` : o.label}
        </button>
      ))}
      {/* Only while there is something to take back, and it says what it
          reverses rather than naming the chord — the write is the surprising
          thing, not the shortcut. */}
      {s.docUndo && !s.rewriting && (
        <button className="s" type="button"
                style={{ height: 32, padding: '0 10px' }}
                title="Put back what you wrote (⌘Z in the prompt)"
                onClick={() => s.undoDoc()}>
          Undo
        </button>
      )}
    </>
  )
}
