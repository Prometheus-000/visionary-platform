import { useCallback, useEffect, useState } from 'react'

import { everyMs, failed } from '../api/client'
import {
  deleteSession, listSessions, putSession, startSession, stopSession,
} from '../api/routes'
import type { Session, TrainParams } from '../api/types'

/**
 * The training board: every card, and what each one's run is doing.
 *
 * **A run is no longer something the page owns.** It used to be one `job` in
 * component state, which is what made "one at a time" true — `train_job` shares
 * nothing between runs and never did, so the limit was a variable rather than
 * the backend. What replaces it is a list of records that outlive the window:
 * close the tab mid-run and the card is still there, with its bar where it got
 * to, when you come back.
 *
 * One poll for the whole board, not one per card. A browser gives an origin
 * about six connections and everything on this page comes off it, so four cards
 * each polling their own job is four of the six spent to say four numbers the
 * server already holds together.
 */
export type Draft = {
  id?: string
  lora_name: string
  trigger_word: string
  dataset: string
  params: Record<string, string | boolean>
}

/** Active means a container is running, or about to be. Nothing else may be
 *  edited, started or deleted without saying what it costs — and everything
 *  else may be, which is the whole of the card's two states. */
export const isActive = (s: Session) => s.status === 'running' || s.status === 'queued'

export function useSessions(open: boolean) {
  const [rows, setRows] = useState<Session[]>([])
  // How many the server holds, which is not always how many it sent: the
  // listing is bounded because it is polled, and a cap that is not reported is
  // a board that looks like the whole board and is not.
  const [total, setTotal] = useState(0)
  const [error, setError] = useState<string | null>(null)
  const [loaded, setLoaded] = useState(false)
  const live = rows.some(isActive)

  const load = useCallback(async () => {
    const r = await listSessions()
    if (failed(r)) return setError(r.error)
    setError(null)
    setLoaded(true)
    setRows(r.sessions ?? [])
    setTotal(r.total ?? (r.sessions ?? []).length)
  }, [])

  useEffect(() => {
    void load()
  }, [load])

  /*
   * Three rates, and the middle one is the reason this is not a constant.
   *
   * 2500ms with the board open: a run is hours long and a card's numbers move
   * about once a second, so the canvas's 400ms would spend the browser's six
   * connections to say nothing new. 15s when Train is closed but something is
   * training, which is all the door's readout needs — and *nothing at all*
   * when Train is closed and nothing is running, because a run can only start
   * from the board, so there is no state to miss. That last case is the common
   * one: Generate should not be polling a board nobody has opened.
   *
   * The dependency is `live` rather than `rows` on purpose. Depending on the
   * list would tear down and rebuild the interval on every reply, which is the
   * overlap `everyMs` exists to prevent, reintroduced one layer up.
   */
  useEffect(() => {
    if (!open && !live) return
    const t = everyMs(load, open ? 2500 : 15_000)
    return () => clearInterval(t)
  }, [open, live, load])

  // Returns the reply rather than a boolean, because the one caller that needs
  // more than "did it work" is Save-and-start: the card it must start is the
  // one the route just wrote, and `rows` in the caller's closure is the list
  // from before the reload. Reading an id out of state that has not been set
  // yet is the bug this shape prevents.
  const act = useCallback(async <T,>(fn: () => Promise<T | { error: string }>) => {
    const r = await fn()
    if (r && typeof r === 'object' && 'error' in r) {
      setError((r as { error: string }).error)
      return null
    }
    setError(null)
    await load()
    return r as T
  }, [load])

  return {
    rows, total, error, loaded, setError,
    load,
    /** Create when the draft carries no id, save an edit when it does. */
    save: (d: Draft) => act(() => putSession(d)),
    start: (id: string) => act(() => startSession(id)),
    stop: (id: string) => act(() => stopSession(id)),
    remove: (id: string) => act(() => deleteSession(id)),
  }
}

/** The dials as the form holds them: strings, because they come out of inputs.
 *  The server casts and clamps, so this never has to decide what "1e-4" means. */
export function paramsToForm(p: Partial<TrainParams> | undefined,
                             defaults: TrainParams): Record<string, string | boolean> {
  const src = { ...defaults, ...(p ?? {}) }
  return Object.fromEntries(
    Object.entries(src).map(([k, v]) => [k, typeof v === 'boolean' ? v : String(v)]),
  )
}
