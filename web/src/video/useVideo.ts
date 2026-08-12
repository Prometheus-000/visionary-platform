import { useCallback, useState } from 'react'

import { everyMs, failed } from '../api/client'
import { status, stop, video } from '../api/routes'
import type { JobStatus } from '../api/types'
import { loraIndex, readVidLoras, stripLoras } from '../lora/tokens'
import { negAllowed, readShot, useStore, type Store } from '../store'
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
  jobId: string | null
  file: string | null
  percent: number
  phase: string
  error: string | null
  meta: string[]
}

const IDLE: VideoRun = {
  running: false, jobId: null, file: null, percent: 0, phase: '', error: null, meta: [],
}

export function videoBody(s: Store): Record<string, unknown> {
  const r = resolveVid(s)
  const index = loraIndex(s.state)
  return {
    model: s.vid.model,
    prompt: stripLoras(s.prompt),
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

export function useVideo(onLanded: () => void) {
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
    setRun({ running: false, jobId, file, percent: 100, phase: '', error: null, meta })
    onLanded()
  }, [onLanded])

  const start = useCallback(async () => {
    const s = useStore.getState()
    if (!stripLoras(s.prompt)) return
    setRun({ ...IDLE, running: true, phase: 'Queued…' })
    const r = await video(videoBody(s))
    if (failed(r)) {
      setRun({ ...IDLE, error: r.error })
      return
    }
    const jobId = r.job_id
    setRun((p) => ({ ...p, jobId }))
    const t = everyMs(async () => {
      const st = await status(jobId)
      if (failed(st)) return
      if (st.status === 'completed') {
        clearInterval(t)
        finish(st, jobId)
      } else if (st.status === 'failed') {
        clearInterval(t)
        setRun({ ...IDLE, error: st.error || 'Generation failed' })
      } else if (st.status === 'stopped') {
        clearInterval(t)
        setRun({ ...IDLE, meta: ['Cancelled.'] })
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
    if (!run.jobId) return
    setRun((p) => ({ ...p, phase: 'Stopping…' }))
    await stop(run.jobId)
  }, [run.jobId])

  /** The canvas only. The prompt, the pills, the boxes and the settings are all still
   *  what you were working on — this clears the result, which is the one thing "clear"
   *  can mean when everything else is an input you are mid-edit of. */
  const clear = useCallback(() => setRun(IDLE), [])

  return { run, start, cancel, clear }
}
