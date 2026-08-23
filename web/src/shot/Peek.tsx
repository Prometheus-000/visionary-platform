import { useEffect, useState } from 'react'

import { failed } from '../api/client'
import { compile } from '../api/routes'
import { readShot, useStore } from '../store'
import { stripLoras } from '../lora/tokens'
import { readScene, typedProse } from '../scene/model'
import { resolveVid } from '../console/resolve'

/**
 * What the model will actually be given.
 *
 * This is the answer to the expensive half of the problem: a take costs two to
 * three minutes, so every question about the format used to be paid for at that
 * rate — and without this the only way to answer "where did my camera direction
 * go" was to render again.
 *
 * Compiled by **the same route that compiles the real run**, on the same CPU
 * container. Never a second implementation in here: a preview with its own
 * implementation is a preview that can disagree with what happens, which is worse
 * than no preview at all. `tools/ui-checks/probe_compile.py` pins 402 of its
 * outputs for exactly this reason.
 */
export function Peek() {
  const s = useStore()
  const [text, setText] = useState('')

  // Offered only when there is something to compile. With no pills the compiled
  // prompt is the typed one, and a disclosure that opens to show you your own
  // sentence back is a control with nothing to say.
  // On the video side the scene is most of the document, and it can be live with
  // no pill on the rail at all — a second shot alone changes everything about
  // what the encoder is handed.
  const sc = s.kind === 'video' ? readScene(s.scene, s.pool) : null
  const has = s.shot.length > 0 || s.refRoles.some(Boolean) || !!sc
  const open = s.peekOpen && has

  useEffect(() => {
    if (!open) return
    // Debounced, because this follows every keystroke in the prompt as well as
    // every pill: the typed sentence is most of the document.
    const t = window.setTimeout(async () => {
      const r = await compile({
        kind: s.kind,
        model: s.vid.model,
        prompt: stripLoras(s.kind === 'video' ? typedProse(s.scene) : s.prompt),
        shot: readShot(s.shot),
        seconds: Number(resolveVid(s).seconds) || undefined,
        ...(sc && { scene: sc.scene }),
        // Never the bytes. This is re-fetched on every pill and every keystroke,
        // and what the compiler needs from a reference is that there is one —
        // the scene's own refs are indices into exactly these counts, so they
        // have to be the counts the run would upload rather than the trays.
        first_frame: !!s.keyframe.first,
        last_frame: !!s.keyframe.last,
        references: sc ? sc.references.length : s.refs.length,
        ref_videos: sc ? sc.ref_videos.length : s.refVids.length,
        ...(sc && { ref_audios: sc.ref_audios.length }),
        ref_roles: sc ? [] : s.refRoles.slice(0, s.refs.length),
      })
      setText(failed(r) ? r.error : r.prompt)
    }, 220)
    return () => window.clearTimeout(t)
  }, [open, s])

  if (!has) return null
  return (
    <div id="shot-peek" className={open ? 'open' : ''}>
      <button type="button" onClick={() => s.setPeekOpen(!s.peekOpen)}>what the model reads</button>
      {open && <pre>{text || '—'}</pre>}
    </div>
  )
}
