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
 * **Two shapes, one line of copy.** On the image side the compiled prompt is
 * genuinely a fold under the sentence above it, so it folds. On the video side it
 * is a six-field document and reading one is not a step in composing, so the same
 * words open `SourcePane` instead — view source, taking the canvas.
 *
 * **It is advertised, and it was not.** The ladder's argument was that devtools is
 * a chord and a context menu and the page never announces it, which is true of
 * devtools and wrong here: a chord nobody is told about is a feature nobody has.
 * ⌘⌥U and the context menu both still work; this is the way in you can see.
 */
export function Peek() {
  const s = useStore()
  const video = s.kind === 'video'

  // Offered only when there is something to compile. With no pills the compiled
  // prompt is the typed one, and a disclosure that opens to show you your own
  // sentence back is a control with nothing to say. On the video side there is
  // always a scene, so there is always a document.
  const has = video || s.shot.length > 0 || s.refRoles.some(Boolean)
  const open = !video && s.peekOpen && has
  const text = useCompiled(open)

  if (!has) return null
  return (
    <div id="shot-peek" className={open ? 'open' : ''}>
      <button type="button"
              onClick={() => { video ? s.setDocOpen(!s.docOpen) : s.setPeekOpen(!s.peekOpen) }}>
        what the model reads
      </button>
      {open && <pre>{text || '—'}</pre>}
    </div>
  )
}
