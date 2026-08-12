import { useCallback, useRef, useState } from 'react'

import { everyMs, failed } from '../api/client'
import { generate, status, stop } from '../api/routes'
import type { JobStatus } from '../api/types'

/**
 * One render, from press to pictures.
 *
 * Polls through `everyMs` at 400ms. That interval is why `everyMs` exists at
 * all: `/api/status` reads a *network* Dict, and `setInterval` fires on a clock
 * rather than on a reply, so a slow answer does not delay the next tick, it
 * overlaps it. Replies then land out of order and the bar is painted by
 * whichever arrived last rather than whichever is newest — step 14 lands, step
 * 12 lands on top of it, and the bar walks backwards.
 */
export type RunState = {
  running: boolean
  jobId: string | null
  percent: number
  phase: string
  error: string | null
  files: string[]
  /** The facts about the render, in the page's order: seeds, then the sampler
   *  line, then how long it took. */
  meta: string[]
  /** LoRAs the run reports it could not apply, with the reason it gave. See
   *  the `=== false` note below. */
  skipped: string[]
}

const IDLE: RunState = {
  running: false, jobId: null, percent: 0, phase: '',
  error: null, files: [], meta: [], skipped: [],
}

export function useGenerate() {
  const [run, setRun] = useState<RunState>(IDLE)
  const timer = useRef<number | null>(null)

  const finish = useCallback((s: JobStatus, jobId: string) => {
    const files = (s.files as string[] | undefined) ?? []

    // `=== false`, not `!l.applied`. The image report emits {name, unet,
    // text_encoder} and no `applied` key at all, so a falsy test called every
    // LoRA on every render unapplied — a warning that is always on is worse
    // than no warning, and this one cost real time during a debug by pointing
    // at a healthy LoRA while the actual fault was elsewhere. Both other
    // readers of this field already default the other way.
    const loras =
      (s.loras as { name?: string; applied?: boolean; reason?: string }[] | undefined) ?? []
    // The reason rides along: "not applied: k3nan (no matching keys)" says
    // which LoRA and why, and the second half is the part that tells a wrong
    // filename apart from a LoRA trained for another architecture.
    const skipped = loras
      .filter((l) => l.applied === false)
      .map((l) => `${String(l.name ?? '')}${l.reason ? ` (${l.reason})` : ''}`)

    // Field names are the record's, not ones that sound right: `seeds` is a
    // list because a batch has one per image, and the duration is `duration_s`.
    const seeds = (s.seeds as number[] | undefined) ?? []
    const meta = [
      seeds.length ? `seed ${seeds.join(', ')}` : '',
      s.sampler ? `${String(s.sampler)} · ${String(s.steps)} steps · CFG ${String(s.cfg_scale)}` : '',
      s.duration_s ? `${String(s.duration_s)}s` : '',
    ].filter(Boolean)

    setRun({
      running: false, jobId, percent: 100, phase: '', error: null, files, meta, skipped,
    })
  }, [])

  const start = useCallback(
    async (payload: Record<string, unknown>) => {
      setRun({ ...IDLE, running: true, phase: 'Queued…' })
      const r = await generate(payload)
      if (failed(r)) {
        setRun({ ...IDLE, error: r.error })
        return
      }
      const jobId = r.job_id
      setRun((s) => ({ ...s, jobId }))

      const t = everyMs(async () => {
        const s = await status(jobId)
        if (failed(s)) return
        if (s.status === 'completed') {
          clearInterval(t)
          finish(s, jobId)
        } else if (s.status === 'failed') {
          clearInterval(t)
          setRun({ ...IDLE, error: s.error || 'Generation failed' })
        } else if (s.status === 'stopped') {
          clearInterval(t)
          setRun({ ...IDLE, meta: ['Cancelled.'] })
        } else {
          setRun((prev) => ({
            ...prev,
            running: true,
            percent: Number(s.percent ?? 0),
            // The step count when there is one: "Working…" is equally true of a
            // run on step 2 and one that has stalled.
            phase: s.step ? `Step ${s.step}/${String(s.steps ?? s.total_steps ?? '?')}` : (s.phase || 'Working…'),
          }))
        }
      }, 400)
      timer.current = t
    },
    [finish],
  )

  /** Cooperative: the job checks a flag between steps and unwinds cleanly, so
   *  the container survives and the next request is warm. */
  const cancel = useCallback(async () => {
    if (!run.jobId) return
    setRun((s) => ({ ...s, phase: 'Stopping…' }))
    await stop(run.jobId)
  }, [run.jobId])

  return { run, start, cancel }
}
