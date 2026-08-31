/**
 * One Playground job, from press to files — and the room's other jobs too.
 *
 * The run, the engine restart, the catalogue harvest and a pack install are
 * all the same shape on the wire (spawn, then poll /api/status), so one hook
 * watches whichever is in flight. Polls through `everyMs` at 400ms for the
 * reason that helper exists — see useGenerate.
 */
import { useCallback, useRef, useState } from 'react'

import { everyMs, failed, type ApiError } from '../api/client'
import {
  installPack, playgroundRefresh, playgroundRestart, playgroundRun, stop,
  status,
} from '../api/routes'
import { useStore } from '../store'

export type PgRun = {
  running: boolean
  /** What the job in flight is — 'run' renders, the others are room chores.
   *  The results strip only repaints for a 'run'. */
  doing: 'run' | 'restart' | 'catalogue' | 'pack' | null
  runId: string | null
  jobId: string | null
  files: string[]
  phase: string
  percent: number
  error: string | ApiError | null
  /** The node that broke, when the server could name one — the editor lights
   *  it up instead of making you read the id out of the message. */
  errorNode: string | null
  /** What a finished chore said — "restart the engine to load it". */
  note: string | null
}

const IDLE: PgRun = {
  running: false, doing: null, runId: null, jobId: null, files: [],
  phase: '', percent: 0, error: null, errorNode: null, note: null,
}

export function usePlayground(onDone?: (doing: PgRun['doing']) => void) {
  const [run, setRun] = useState<PgRun>(IDLE)
  const timer = useRef<number | null>(null)

  const watch = useCallback((runId: string, doing: PgRun['doing']) => {
    if (timer.current !== null) clearInterval(timer.current)
    setRun((p) => ({ ...p, running: true, doing, runId, error: null,
                     errorNode: null, note: null, phase: 'Queued…', percent: 0 }))
    const t = everyMs(async () => {
      const st = await status(runId)
      if (failed(st)) return
      if (st.status === 'completed') {
        clearInterval(t)
        setRun((p) => ({
          ...p, running: false, doing: null, runId: null, phase: '',
          percent: 0, note: st.note ?? null,
          ...(doing === 'run'
            ? { jobId: runId, files: st.files ?? [] } : {}),
        }))
        onDone?.(doing)
      } else if (st.status === 'failed') {
        clearInterval(t)
        setRun((p) => ({
          ...p, running: false, doing: null, runId: null, phase: '',
          error: st.error || 'The job failed and gave no reason.',
          errorNode: st.error_node ?? null,
        }))
      } else if (st.status === 'stopped') {
        clearInterval(t)
        setRun((p) => ({ ...p, running: false, doing: null, runId: null,
                         phase: '' }))
      } else {
        setRun((p) => ({
          ...p, running: true,
          percent: Number(st.percent ?? 0),
          phase: st.step
            ? `Step ${st.step}/${String(st.total_steps ?? st.steps ?? '?')}`
            : (st.phase === 'loading' ? 'Loading the model…'
               : (st.phase || 'Working…')),
        }))
      }
    }, 400)
    timer.current = t
  }, [onDone])

  const fire = useCallback(async () => {
    const s = useStore.getState()
    if (!s.pg.graph) return
    const r = await playgroundRun({
      graph: s.pg.graph,
      host: s.pg.host,
      attachments: s.pg.attachments,
      workflow_name: s.pg.name || undefined,
      gpu: s.gpu || undefined,
    })
    if (failed(r)) {
      setRun((p) => ({ ...p, error: r, errorNode: null }))
      return
    }
    watch(r.job_id, 'run')
  }, [watch])

  const cancel = useCallback(async () => {
    const id = run.runId
    if (id) await stop(id)
  }, [run.runId])

  const restart = useCallback(async () => {
    const r = await playgroundRestart(useStore.getState().pg.host)
    if (failed(r)) { setRun((p) => ({ ...p, error: r })); return }
    watch(r.job_id, 'restart')
  }, [watch])

  const harvest = useCallback(async () => {
    const r = await playgroundRefresh()
    if (failed(r)) { setRun((p) => ({ ...p, error: r })); return }
    watch(r.job_id, 'catalogue')
  }, [watch])

  const install = useCallback(async (url: string, ref?: string) => {
    const r = await installPack(url, ref || undefined)
    if (failed(r)) { setRun((p) => ({ ...p, error: r })); return }
    watch(r.job_id, 'pack')
  }, [watch])

  const clearError = useCallback(() => {
    setRun((p) => ({ ...p, error: null, errorNode: null, note: null }))
  }, [])

  return { run, fire, cancel, restart, harvest, install, clearError }
}
