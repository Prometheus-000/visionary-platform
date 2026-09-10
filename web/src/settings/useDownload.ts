import { useCallback, useRef, useState } from 'react'

import { everyMs, failed, type Res } from '../api/client'
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
  /** What the server actually said, when there was a server behind the failure. Only a
   *  refused *start* has one: `begin`'s call answers `{error, detail}` and the detail is
   *  the traceback or the fetch error underneath the sentence. A job that fails while
   *  running carries only `s.error`, so this stays absent there rather than being filled
   *  with something invented. `Line` folds it into the err-box's disclosure. */
  detail?: string
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
          // The job's own sentence when it wrote one, because it knows something
          // `doneText` cannot: a weight that was already on the volume, or the
          // three files of five a Drive folder did not have to fetch. "Downloaded."
          // over the top of that throws away the only evidence the diff did anything.
          const note = typeof s.note === 'string' ? s.note : ''
          set(id, { percent: 100, message: note || doneText, tone: 'ok', running: false })
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
          // The fallback is reached when the record is failed with no error on it — a
          // container killed mid-pull, or one that ran out of disk, never gets as far as
          // writing one. It said "Download failed", which names no next step at all, so
          // the only move it suggested was closing the sheet. Two things *are* known
          // here: how many files landed, and that clearing `busy` hands the uplink back,
          // so pressing Download again is a real option rather than a hope. The same
          // count the Cancel branch above prints, for the same reason — a queue that got
          // four files in has done work worth naming.
          const got = (s.downloaded as unknown[] | undefined)?.length ?? 0
          set(id, {
            message: s.error || (got
              ? `Download failed after ${got} file${got === 1 ? '' : 's'} — those are on the`
                + ' volume and are not fetched twice, so press Download again to pick up the rest.'
              : 'Download failed before any file landed — press Download again; if it stops the'
                + ' same way, a gated repo needs the HuggingFace token saved above.'),
            tone: 'err', running: false,
          })
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
           // `Res<DownloadStart>`, not `DownloadStart | {error: string}`: the literal
           // stopped matching `ApiError` when it gained an optional `detail`, and the
           // failed() branch then narrowed nothing — every field read below it was
           // suddenly being read off the error type too.
           call: () => Promise<Res<DownloadStart>>,
           onDone: () => void) => {
      setBusy(true)
      // `detail: undefined` explicitly: `set` merges into the row's last state, so a
      // traceback from the attempt before this one would otherwise still be sitting in
      // the disclosure under a message it has nothing to do with.
      set(id, { percent: 0, message: 'Starting…', tone: 'plain', running: true, detail: undefined })
      const r = await call()
      if (failed(r)) {
        set(id, { message: r.error, detail: r.detail, tone: 'err', running: false })
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
