import { useStore } from '../store'
import { useCompiled } from './useCompiled'

/**
 * What the model will actually be given.
 *
 * This is the answer to the expensive half of the problem: a take costs two to
 * three minutes, so every question about the format used to be paid for at that
 * rate, and without it the only way to answer "where did my camera direction go"
 * was to render again.
 *
 * **The caret folds, on both sides — because a caret is a promise.** The video
 * side used to route this same label to `SourcePane`, a surface that takes the
 * console, and the owner's correction is the reason it stopped: a caret means
 * the content expands underneath, and a control whose gesture delivers a
 * different pattern than its glyph promises is the page lying in a small way.
 * So the fold is the read path everywhere. `SourcePane` survives as the *edit*
 * path only — ⌘⌥U or right-click, where detaching the document is the point.
 */
export function Peek() {
  const s = useStore()
  const video = s.kind === 'video'

  // Offered only when there is something to compile. With no pills the compiled
  // prompt is the typed one, and a disclosure that opens to show you your own
  // sentence back is a control with nothing to say. On the video side there is
  // always a scene, so there is always a document.
  const has = video || s.shot.length > 0 || s.refRoles.some(Boolean)
  const open = s.peekOpen && has
  const text = useCompiled(open)

  if (!has) return null
  return (
    <div id="shot-peek" className={open ? 'open' : ''}>
      <button type="button" onClick={() => { s.setPeekOpen(!s.peekOpen) }}>
        what the model reads
      </button>
      {open && <pre>{text || '—'}</pre>}
    </div>
  )
}
