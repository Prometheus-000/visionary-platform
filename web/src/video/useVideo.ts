import { useCallback, useState } from 'react'

import { everyMs, failed } from '../api/client'
import { status, stop, video } from '../api/routes'
import type { JobStatus } from '../api/types'
import type { GalleryItem } from '../gallery/types'
import { loraIndex, readVidLoras, stripLoras } from '../lora/tokens'
import { docFor, docFrom, negAllowed, readShot, useStore, type Store } from '../store'
import { resolveVid } from '../console/resolve'

/**
 * One clip, from press to playback.
 *
 * The same job/status/stop contract the image side uses, at the same 400ms — adding
 * Wan did not add a backend and neither does this: what is per-family is a graph
 * builder on the server and a row in `VIDEO_MODELS`.
 *
 * The one thing that differs from `useGenerate` is what the phase says while nothing
 * is sampling yet. The first minutes of a video run are 42.5 GB loading onto the card,
 * with no step count to show — naming that beats a bar that sits at zero looking stuck.
 */
export type VideoRun = {
  running: boolean
  /** The clip on screen — the last run that *completed*. Not cleared when the next
   *  one starts: a new clip replaces the old one when it lands, not when it is asked
   *  for. It matters more here than on the image side, because a take is two to three
   *  minutes and blanking the canvas means the thing you were judging is gone for all
   *  of them. `finish` is the only thing that moves it. */
  jobId: string | null
  file: string | null
  /** The job being polled right now, which is *not* `jobId` until it completes. Two
   *  ids because the bytes on screen and the work in flight are two different clips
   *  during a run. */
  runId: string | null
  percent: number
  phase: string
  error: string | null
  meta: string[]
}

const IDLE: VideoRun = {
  running: false, jobId: null, file: null, runId: null, percent: 0, phase: '',
  error: null, meta: [],
}

export function videoBody(s: Store): Record<string, unknown> {
  const r = resolveVid(s)
  const index = loraIndex(s.state)
  // See `imageBody`: one string, keying both halves of the same request.
  const prompt = stripLoras(s.prompt)
  return {
    model: s.vid.model,
    prompt,
    modules: docFor(s, prompt),
    // See `imageBody` — what they wrote before the replacement was written.
    prompt_original: docFrom(s, prompt),
    negative_prompt: negAllowed(s) ? s.negative : '',
    aspect: s.vid.aspect,
    tier: r.tier,
    seconds: r.seconds,
    steps: s.vid.steps,
    seed: s.vid.seed,
    cfg: s.vid.cfg,
    shift: s.vid.shift,
    switch_at: s.vid.switchAt,
    sampler: r.sampler,
    scheduler: r.scheduler,
    loras: readVidLoras(index, s.prompt, s.state?.max_loras ?? 6,
                        s.state?.wan_experts ?? ['both', 'high', 'low']),
    shot: readShot(s.shot),
    ref_roles: s.refRoles.slice(0, s.refs.length),
    first_frame: s.keyframe.first,
    last_frame: s.keyframe.last,
    references: s.refs,
    ref_videos: s.refVids,
    ref_size: s.vid.refSize,
    gpu: s.gpu.video,
  }
}

export function useVideo(onLanded: (it: GalleryItem) => void) {
  const [run, setRun] = useState<VideoRun>(IDLE)

  const finish = useCallback((st: JobStatus, jobId: string) => {
    const file = (st.files as string[] | undefined)?.[0] ?? null
    const meta = [
      st.width ? `${String(st.width)}×${String(st.height)}` : '',
      st.seconds ? `${String(st.seconds)}s · ${String(st.frames)} frames · ${String(st.fps)} fps` : '',
      st.seed != null ? `seed ${String(st.seed)}` : '',
      st.steps ? `${String(st.steps)} steps` : '',
      st.duration_s ? `${String(st.duration_s)}s` : '',
    ].filter(Boolean)
    // The pin — see `useGenerate.finish` for why `doc &&` is the whole condition.
    // `s.img.seed` and `s.vid.seed` are separate fields written by their own
    // `finish`, so switching image ↔ video carries no pin across and there is
    // nothing to clear for it.
    const store = useStore.getState()
    if (store.doc && !store.vid.seed && st.seed != null) {
      store.setVid({ seed: String(st.seed) })
    }

    // Atomically, so there is never a frame pairing the old jobId with the new file.
    setRun((p) => ({
      ...p, running: false, jobId, file, runId: null, percent: 100, phase: '',
      error: null, meta,
    }))
    // See useGenerate: the run reports itself rather than the page re-asking the
    // volume about work it just watched finish.
    if (file) {
      onLanded({ ...(st as Partial<GalleryItem>), job_id: jobId, kind: 'video',
                 files: [file], created: Date.now() / 1000 })
    }
  }, [onLanded])

  const start = useCallback(async () => {
    const s = useStore.getState()
    if (!stripLoras(s.prompt)) return
    // Keep the last clip on screen and overlay a progress state on it — see `jobId`
    // above. A cold first run has nothing to keep and shows the full placeholder.
    setRun((p) => ({
      ...p, running: true, runId: null, percent: 0, phase: 'Queued…', error: null,
    }))
    const r = await video(videoBody(s))
    if (failed(r)) {
      // The last clip stays: a request that never started should not blank what you
      // were watching.
      setRun((p) => ({ ...p, running: false, runId: null, error: r.error }))
      return
    }
    const runId = r.job_id
    setRun((p) => ({ ...p, runId }))
    const t = everyMs(async () => {
      const st = await status(runId)
      if (failed(st)) return
      if (st.status === 'completed') {
        clearInterval(t)
        finish(st, runId)
      } else if (st.status === 'failed') {
        clearInterval(t)
        setRun((p) => ({
          ...p, running: false, runId: null, error: st.error || 'Generation failed',
        }))
      } else if (st.status === 'stopped') {
        clearInterval(t)
        // The previous clip coming back is the feedback — there is nothing to say
        // that the returned picture does not already say.
        setRun((p) => ({ ...p, running: false, runId: null, phase: '' }))
      } else {
        setRun((p) => ({
          ...p,
          running: true,
          percent: Number(st.percent ?? 0),
          phase: st.step
            ? `Step ${st.step}/${String(st.total_steps ?? st.steps ?? '?')}`
              + (st.eta ? ` · ${String(st.eta)} left` : '')
            : (st.phase === 'loading' ? 'Loading the model…' : (st.phase || 'Working…')),
        }))
      }
    }, 400)
  }, [finish])

  const cancel = useCallback(async () => {
    if (!run.runId) return
    setRun((p) => ({ ...p, phase: 'Stopping…' }))
    await stop(run.runId)
  }, [run.runId])

  /** The canvas only. The prompt, the pills, the boxes and the settings are all still
   *  what you were working on — this clears the result, which is the one thing "clear"
   *  can mean when everything else is an input you are mid-edit of. */
  const clear = useCallback(() => setRun(IDLE), [])

  return { run, start, cancel, clear }
}
