import { useCallback, useState } from 'react'

import { everyMs, failed, type ApiError } from '../api/client'
import { status, stop, video } from '../api/routes'
import type { JobStatus } from '../api/types'
import type { GalleryItem } from '../gallery/types'
import { readVidChips, stripLoras } from '../lora/tokens'
import { negAllowed, readShot, supports, useStore, type Store } from '../store'
import { readScene, typedProse } from '../scene/model'
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
  /** See `RunState.error`: the whole `ApiError`, so `ErrorNote` still has the
   *  server's report to fold away under the sentence. */
  error: string | ApiError | null
  meta: string[]
}

const IDLE: VideoRun = {
  running: false, jobId: null, file: null, runId: null, percent: 0, phase: '',
  error: null, meta: [],
}

export function videoBody(s: Store): Record<string, unknown> {
  const r = resolveVid(s)
  // See `imageBody`: one string, keying both halves of the same request.
  const prompt = stripLoras(typedProse(s.scene))
  // Null when the scene is not live, and that is the contract rather than an
  // optimisation: no cast, one shot and nothing chosen, and the run is the typed
  // sentence byte-for-byte — the same document a prompt box would have produced,
  // because there is no document. Its three asset lists are what `<Picture N>`
  // numbers against, so they replace the flat trays whenever it is live.
  const sc = readScene(s.scene, s.pool)
  return {
    ...(sc && { scene: sc.scene }),
    // **The edit is what runs.** A document taken over by hand travels in its own
    // field rather than as `prompt`, because `prompt_typed` is the prose somebody
    // wrote and only the receipt is being overridden — folded together, Reuse
    // would load a six-field schema into the first shot's row and compile *that*
    // on the next run. See `SourcePane`.
    ...(s.doc !== null && { prompt_compiled: s.doc }),
    model: s.vid.model,
    prompt,
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
    // The expert list is per *model*, not global: `wan_experts` describes the
    // A14B pair, and handing it to a one-expert model invites a tag that model
    // has no branch for.
    loras: readVidChips(s.loras, s.state?.max_loras ?? 6,
                        supports(s).experts
                          ? (s.state?.wan_experts ?? ['both', 'high', 'low'])
                          : ['both']),
    shot: readShot(s.shot),
    ref_roles: sc ? [] : s.refRoles.slice(0, s.refs.length),
    first_frame: s.keyframe.first,
    last_frame: s.keyframe.last,
    // The cast's files when there is a cast, and the flat trays otherwise. Never
    // both: `<Picture N>` is a *position* in this array, so a cast ref pointing
    // at index 1 and a tray photo also sitting at index 1 is a well-formed
    // document naming somebody else's face.
    references: sc ? sc.references : s.refs,
    ref_videos: sc ? sc.ref_videos : s.refVids,
    ...(sc && { ref_audios: sc.ref_audios }),
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
    if (!stripLoras(typedProse(s.scene))) return
    // Keep the last clip on screen and overlay a progress state on it — see `jobId`
    // above. A cold first run has nothing to keep and shows the full placeholder.
    setRun((p) => ({
      ...p, running: true, runId: null, percent: 0, phase: 'Queued…', error: null,
    }))
    const r = await video(videoBody(s))
    if (failed(r)) {
      // The last clip stays: a request that never started should not blank what you
      // were watching. The whole `ApiError` — see `useGenerate`, same reason.
      setRun((p) => ({ ...p, running: false, runId: null, error: r }))
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
        // See `useGenerate` for why the bare fallback went. The advice differs on
        // this side because the failures do: a clip is the run that dies on card
        // memory, and duration is the one lever in the strip that changes how much
        // of it the run asks for.
        setRun((p) => ({
          ...p, running: false, runId: null,
          error: st.error
            || 'The clip failed and the job gave no reason — press Generate to try'
               + ' again, or pick a shorter duration if it keeps failing.',
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
