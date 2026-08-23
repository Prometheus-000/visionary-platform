import { useStore } from '../store'
import { useCompiled } from './useCompiled'

/**
 * What the model will actually be given — the image side's disclosure.
 *
 * This is the answer to the expensive half of the problem: a take costs two to
 * three minutes, so every question about the format used to be paid for at that
 * rate, and without it the only way to answer "where did my camera direction go"
 * was to render again.
 *
 * **The video side has no disclosure any more.** Its document is a scene rather
 * than a sentence with clauses stapled to it, and reading one is not a step in
 * composing — so it is view source there, reached by a chord and a context menu
 * and advertised by nothing. See `SourcePane`. This stays where the compiled
 * prompt genuinely *is* a fold under the sentence above it.
 */
export function Peek() {
  const s = useStore()

  // Offered only when there is something to compile. With no pills the compiled
  // prompt is the typed one, and a disclosure that opens to show you your own
  // sentence back is a control with nothing to say.
  const has = s.kind === 'image' && (s.shot.length > 0 || s.refRoles.some(Boolean))
  const open = s.peekOpen && has
  const text = useCompiled(open)

  if (!has) return null
  return (
    <div id="shot-peek" className={open ? 'open' : ''}>
      <button type="button" onClick={() => s.setPeekOpen(!s.peekOpen)}>what the model reads</button>
      {open && <pre>{text || '—'}</pre>}
    </div>
  )
}
