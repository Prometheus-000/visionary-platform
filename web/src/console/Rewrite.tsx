import { failed } from '../api/client'
import { rewrite } from '../api/routes'
import { stripLoras } from '../lora/tokens'
import { useStore } from '../store'

/**
 * Three jobs the interpreter will do on your sentence, and nothing else.
 *
 * **This replaces the underlines rather than sitting beside them.** The
 * semantic layer marked what the model supplied so you could tell your words
 * from its own; measured, that apparatus refused every prompt worth having and
 * the marks it did paint were computed from a provenance claim the model got
 * wrong every time it made one. The trust question it was answering is answered
 * here by the gesture instead: you chose Expand, so you already know why the
 * sentence got longer, and ⌘Z takes it back.
 *
 * The row costs nothing until there is something to act on — the same rule
 * `#shot-rail:empty` encodes, and the reason this is three buttons rather than
 * one menu: a row is affordable when it carries content and never when it
 * carries a single control.
 */
export function Rewrite() {
  const s = useStore()
  const ops = s.state?.rewrite_ops ?? []
  const prose = stripLoras(s.prompt).trim()
  if (!ops.length || !prose) return null

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
    <div id="rewrite-rail">
      {ops.map((o) => (
        <button key={o.key} type="button" className="opt rw"
                disabled={!!s.rewriting}
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
        <button type="button" className="opt rw undo"
                title="Put back what you wrote (⌘Z in the prompt)"
                onClick={() => s.undoDoc()}>
          Undo
        </button>
      )}
    </div>
  )
}
