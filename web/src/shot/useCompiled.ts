import { useEffect, useState } from 'react'

import { failed } from '../api/client'
import { compile } from '../api/routes'
import { resolveVid } from '../console/resolve'
import { stripLoras } from '../lora/tokens'
import { readScene, typedProse } from '../scene/model'
import { readShot, useStore } from '../store'

/**
 * The exact document a run would be given, refetched as you compose.
 *
 * **The one route the React side must never reimplement.** It is the same
 * compiler on the same CPU container as the run, which is what makes both
 * surfaces built on it truthful — a preview with its own implementation is a
 * preview that can disagree with what happens, which is worse than no preview.
 * `tools/ui-checks/probe_compile.py` pins 402 of its outputs for that reason.
 *
 * Two callers now, which is why it left `Peek`: the image side's disclosure and
 * the video side's source pane. Debounced, because it follows every keystroke as
 * well as every pill — the prose is most of the document.
 *
 * Never the bytes. This is re-fetched four times a second while somebody types,
 * and what the compiler needs from a reference is only that there *is* one.
 */
export function useCompiled(active: boolean): string {
  const s = useStore()
  const [text, setText] = useState('')

  useEffect(() => {
    if (!active) return
    const t = window.setTimeout(async () => {
      const sc = s.kind === 'video' ? readScene(s.scene, s.pool) : null
      const r = await compile({
        kind: s.kind,
        model: s.vid.model,
        prompt: stripLoras(s.kind === 'video' ? typedProse(s.scene) : s.prompt),
        shot: readShot(s.shot),
        seconds: Number(resolveVid(s).seconds) || undefined,
        ...(sc && { scene: sc.scene }),
        first_frame: !!s.keyframe.first,
        last_frame: !!s.keyframe.last,
        // The scene's own refs are indices into exactly these counts, so they
        // have to be what the run would upload rather than what is in the trays.
        references: sc ? sc.references.length : s.refs.length,
        ref_videos: sc ? sc.ref_videos.length : s.refVids.length,
        ...(sc && { ref_audios: sc.ref_audios.length }),
        ref_roles: sc ? [] : s.refRoles.slice(0, s.refs.length),
      })
      setText(failed(r) ? r.error : r.prompt)
    }, 220)
    return () => { window.clearTimeout(t) }
  }, [active, s])

  return text
}
