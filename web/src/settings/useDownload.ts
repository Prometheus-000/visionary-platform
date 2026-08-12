import { useCallback, useRef, useState } from 'react'

import { everyMs, failed } from '../api/client'
import { status, stop, type DownloadStart } from '../api/routes'

/**
 * Follow one download to a terminal state.
 *
 * One loop for both shapes. "This weight" and "this family's missing weights"
 * differ only in which keys the server put in the queue — same record, same
 * phases, same Cancel — so they get one follower rather than one each.
 *
 * Polls through `everyMs`, never `setInterval`: a poll waits for its own reply.
 * These land against a *network* Dict, and a status that overlaps its
 * predecessor paints the bar with whichever reply arrived last rather than
 * whichever is newest.
 *
 * 3000ms, matching the page. A download is minutes long and its own container
 * publishes bytes; polling it at the generate loop's 400ms would spend the
 * browser's six connections on a progress bar.
 */
export type Progress = {
  percent: number
  /** The rate, not just the phase. "Downloading…" is equally true of a transfer
   *  moving at 240 MB/s and one that has stalled, which is the state this whole
   *  readout exists to tell apart. */
  message: string
  tone: 'plain' | 'ok' | 'err'
  running: boolean
}

const IDLE: Progress = { percent: 0, message: '', tone: 'plain', running: false }

export function useDownload() {
  const [byId, setById] = useState<Record<string, Progress>>({})
  /** True while any download holds the uplink. Every button reads it. */
  const [busy, setBusy] = useState(false)
  const timers = useRef<Record<string, number>>({})

  const set = useCallback((id: string, p: Partial<Progress>) => {
    setById((m) => ({ ...m, [id]: { ...(m[id] ?? IDLE), ...p } }))
  }, [])

  const follow = useCallback(
    (id: string, jobId: string, doneText: string, onDone: () => void) => {
      const t = everyMs(async () => {
        const s = await status(jobId)
        if (failed(s)) return
        const pct = Number(s.percent ?? 0)
        const rate = s.mb_s ? `${String(s.mb_s)} MB/s` : ''
        if (s.status === 'completed') {
          clearInterval(t)
          set(id, { percent: 100, message: doneText, tone: 'ok', running: false })
          setBusy(false)
          onDone()
        } else if (s.status === 'stopped') {
          clearInterval(t)
          // Names what did land. A queue stopped four files in has done real
          // work, and "Cancelled." alone throws away the only record of which.
          const got = (s.downloaded as unknown[] | undefined)?.length ?? 0
          const left = (s.remaining as unknown[] | undefined)?.length ?? 0
          set(id, {
            message: got ? `Cancelled — ${got} downloaded, ${left} not` : 'Cancelled.',
            tone: 'plain', running: false,
          })
          setBusy(false)
        } else if (s.status === 'failed') {
          clearInterval(t)
          set(id, { message: s.error || 'Download failed', tone: 'err', running: false })
          setBusy(false)
        } else {
          set(id, {
            percent: pct,
            message: [s.phase || 'Downloading…', rate].filter(Boolean).join(' · '),
            tone: 'plain', running: true,
          })
        }
      }, 3000)
      timers.current[id] = t
    },
    [set],
  )

  /**
   * Start one, and read the answer the way the route means it.
   *
   * An unknown key is still an error. Being busy is not: `mine: false` means
   * something else holds the uplink, which the disabled buttons should have
   * prevented — so it is a plain sentence on that row, not a red box.
   */
  const begin = useCallback(
    async (id: string, jobId: string, doneText: string,
           call: () => Promise<DownloadStart | { error: string }>,
           onDone: () => void) => {
      setBusy(true)
      set(id, { percent: 0, message: 'Starting…', tone: 'plain', running: true })
      const r = await call()
      if (failed(r)) {
        set(id, { message: r.error, tone: 'err', running: false })
        setBusy(false)
        return
      }
      if (!r.job_id) {
        set(id, { message: r.note || 'Nothing missing.', tone: 'plain', running: false })
        setBusy(false)
        onDone()
        return
      }
      if (r.mine === false) {
        set(id, {
          message: `${r.busy_with || 'Another download'} is running.`,
          tone: 'plain', running: false,
        })
        setBusy(false)
        return
      }
      follow(id, r.job_id || jobId, doneText, onDone)
    },
    [follow, set],
  )

  /** Cooperative, like every stop here: the job unwinds between files. */
  const cancel = useCallback(async (id: string, jobId: string) => {
    set(id, { message: 'Stopping…' })
    await stop(jobId)
  }, [])

  return { byId, busy, begin, cancel, progressOf: (id: string) => byId[id] ?? IDLE }
}
