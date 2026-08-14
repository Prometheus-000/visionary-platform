import { useCallback, useState } from 'react'

import { everyMs, failed } from '../api/client'
import { generate, status, stop } from '../api/routes'
import type { JobStatus } from '../api/types'
import type { GalleryItem } from '../gallery/types'
import { loraIndex, readLoras, stripLoras } from '../lora/tokens'
import { readRegions } from '../regions/geometry'
import { attached, negAllowed, readShot, regionsLive, useStore, type Store } from '../store'
import { readSize } from '../console/size'

/**
 * One render, from press to pictures.
 *
 * Polls through `everyMs` at 400ms. That interval is why `everyMs` exists at all:
 * `/api/status` reads a *network* Dict, and `setInterval` fires on a clock rather
 * than on a reply, so a slow answer does not delay the next tick, it overlaps it.
 * Replies then land out of order and the bar is painted by whichever arrived last
 * rather than whichever is newest — step 14 lands, step 12 lands on top of it, and
 * the bar walks backwards.
 */
export type RunState = {
  running: boolean
  /** The render on screen — the last one that *completed*. It is deliberately not
   *  cleared when the next run starts: a new render replaces the old one when it
   *  lands, not when it is asked for, so an iteration you are judging stays up
   *  through the two minutes it takes to make its successor. `finish` is the only
   *  thing that moves it. */
  jobId: string | null
  files: string[]
  /** The job being polled right now, which is *not* `jobId` until it completes.
   *  Two ids because the bytes on screen and the work in flight are two different
   *  renders during a run — `fileUrl(jobId, f)` has to keep addressing the old
   *  job's files while `runId` is the one `/api/status` is asked about. */
  runId: string | null
  percent: number
  phase: string
  error: string | null
  /** The facts about the render, in the page's order: seeds, then the sampler line,
   *  then how long it took. */
  meta: string[]
  /** LoRAs the run reports it could not apply, with the reason it gave. */
  skipped: string[]
}

const IDLE: RunState = {
  running: false, jobId: null, files: [], runId: null, percent: 0, phase: '',
  error: null, meta: [], skipped: [],
}

/**
 * The `/api/generate` body.
 *
 * One function, so the Generate button, ⌘Enter and anything Phase 6 grows later all
 * send the same request — which is what lets the console be *dissolvable*: Phase 6
 * re-hosts these controls onto the canvas and the payload does not notice.
 *
 * Empty strings are passed through rather than coerced: the route's `num()` treats
 * `''` as "use the default" and would take a `0` literally, so turning a blank steps
 * field into a number here would silently ask for zero steps.
 */
export function imageBody(s: Store): Record<string, unknown> {
  const index = loraIndex(s.state)
  const [width, height] = readSize(s.img)
  const regions = readRegions(index, s.regions, regionsLive(s))
  return {
    prompt: stripLoras(s.prompt),
    negative_prompt: negAllowed(s) ? s.negative : '',
    model: s.img.model,
    shot: readShot(s.shot),
    loras: readLoras(index, s.prompt, s.state?.max_loras ?? 6),
    regions,
    region_weight: s.regionWeight,
    // Only when there are boxes to compose around — the backend rejects a plate
    // without regions, and sending one anyway would turn a hidden tile that still
    // holds an image into an error nobody could see the cause of.
    scene: regions.length ? attached(s.frame, 'scene') : null,
    outfit: regions.length ? attached(s.frame, 'outfit') : null,
    width,
    height,
    num_images: s.img.n,
    seed: s.img.seed,
    sampler: s.img.sampler,
    scheduler: s.img.scheduler,
    steps: s.img.steps,
    cfg_scale: s.img.cfg,
    shift: s.img.shift,
    gpu: s.gpu.image,
  }
}

export function useGenerate(onLanded: (it: GalleryItem) => void) {
  const [run, setRun] = useState<RunState>(IDLE)

  const finish = useCallback((s: JobStatus, jobId: string) => {
    const files = (s.files as string[] | undefined) ?? []

    // `=== false`, not `!l.applied`. The image report emits {name, unet,
    // text_encoder} and no `applied` key at all, so a falsy test called every LoRA on
    // every render unapplied — a warning that is always on is worse than no warning,
    // and this one cost real time during a debug by pointing at a healthy LoRA while
    // the actual fault was elsewhere. Both other readers of this field already
    // default the other way.
    const loras =
      (s.loras as { name?: string; applied?: boolean; reason?: string }[] | undefined) ?? []
    // The reason rides along: "not applied: k3nan (no matching keys)" says which LoRA
    // and why, and the second half is the part that tells a wrong filename apart from
    // a LoRA trained for another architecture.
    const skipped = loras
      .filter((l) => l.applied === false)
      .map((l) => `${String(l.name ?? '')}${l.reason ? ` (${l.reason})` : ''}`)

    // Field names are the record's, not ones that sound right: `seeds` is a list
    // because a batch has one per image, and the duration is `duration_s`.
    const seeds = (s.seeds as number[] | undefined) ?? []
    const meta = [
      seeds.length ? `seed ${seeds.join(', ')}` : '',
      s.sampler ? `${String(s.sampler)} · ${String(s.steps)} steps · CFG ${String(s.cfg_scale)}` : '',
      s.duration_s ? `${String(s.duration_s)}s` : '',
    ].filter(Boolean)

    // Replaces the on-screen render atomically: the new job's id and files land in
    // the same set as `running:false`, so there is never a frame where the old
    // `jobId` is paired with the new `files`.
    setRun((p) => ({
      ...p, running: false, jobId, runId: null, files, percent: 100, phase: '',
      error: null, meta, skipped,
    }))
    // The finished run, handed over rather than thrown away. The gallery used to ask
    // the volume to tell it about a job this function was holding in its hand, and
    // believe the answer when it came back without one. `**s` first so the four facts
    // below win — the record is the run's report, and these are what the card draws.
    onLanded({ ...(s as Partial<GalleryItem>), job_id: jobId, kind: 'image', files,
               created: Date.now() / 1000 })
  }, [onLanded])

  const start = useCallback(async () => {
    const s = useStore.getState()
    const body = imageBody(s)
    if (!body.prompt && !(body.regions as unknown[]).length) return
    // Keep the previous render (jobId/files/meta) on screen and only overlay a
    // progress state on it — see the `jobId` note above. A cold first run has no
    // previous render, so this shows the full placeholder; an iteration keeps the
    // last picture up until its replacement is ready.
    setRun((p) => ({
      ...p, running: true, runId: null, percent: 0, phase: 'Queued…', error: null,
    }))
    const r = await generate(body)
    if (failed(r)) {
      // Leave the last render up; a failed request should not blank what you were
      // iterating on. Only the error and the stopped-running state change.
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
        setRun((p) => ({ ...p, running: false, runId: null, error: st.error || 'Generation failed' }))
      } else if (st.status === 'stopped') {
        clearInterval(t)
        // The previous render coming back is the feedback now — a stopped run leaves
        // the picture it was replacing exactly where it was, so there is nothing to
        // say that the returned image does not already say.
        setRun((p) => ({ ...p, running: false, runId: null, phase: '' }))
      } else {
        setRun((prev) => ({
          ...prev,
          running: true,
          percent: Number(st.percent ?? 0),
          // The step count when there is one: "Working…" is equally true of a run on
          // step 2 and one that has stalled.
          phase: st.step
            ? `Step ${st.step}/${String(st.total_steps ?? st.steps ?? '?')}`
            : (st.phase || 'Working…'),
        }))
      }
    }, 400)
  }, [finish])

  /** Cooperative: the job checks a flag between steps and unwinds cleanly, so the
   *  container survives and the next request is warm. Disabled rather than removed on
   *  press, because a Stop that vanishes on click reads as a click that missed. */
  const cancel = useCallback(async () => {
    if (!run.runId) return
    setRun((p) => ({ ...p, phase: 'Stopping…' }))
    await stop(run.runId)
  }, [run.runId])

  const clear = useCallback(() => setRun(IDLE), [])

  return { run, start, cancel, clear }
}
