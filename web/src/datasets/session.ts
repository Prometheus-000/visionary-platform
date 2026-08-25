/**
 * A draft belongs to the window that made it.
 *
 * `sessionStorage`, not `localStorage`, because it survives a reload and dies with the
 * tab — which is exactly the lifetime an unsaved thing should have. The heartbeat is what
 * the server reads as "still open"; without it a draft is swept once the grace period
 * passes. There is no server-side "app closed" event to use instead: the web container
 * scales to zero on Modal's schedule, not yours, so a cold start would be a lifecycle
 * signal that means nothing about whether you are still working.
 */
import { beat as ping } from '../api/routes'

export const SESSION = (() => {
  let s: string | null = null
  try {
    s = sessionStorage.getItem('vis-session')
  } catch { /* private mode, or storage disabled */ }
  if (!s) {
    s = `s${Math.random().toString(36).slice(2, 10)}${Date.now().toString(36)}`
    try {
      sessionStorage.setItem('vis-session', s)
    } catch { /* the id still works for this page's lifetime */ }
  }
  return s
})()

/**
 * The set currently on screen, module-level for the same reason SESSION is:
 * the interval in `keepAlive` outlives any one render, and a beat that named
 * a stale set would adopt a draft the person already left.
 *
 * Why the beat names it at all: a draft records the session that *created* it,
 * and that id dies with its tab. A draft reopened in a later window was being
 * kept alive by a marker nobody touched any more — the sweep deleted 80 images
 * out from under a live curating session. The server adopts whatever draft the
 * beat says is open, so looking at a set is what keeps it alive.
 */
let liveSet: string | null = null

export const setLiveSet = (name: string | null) => {
  liveSet = name
}

export const beat = () => ping(SESSION, liveSet)

/**
 * Only while there is something to keep alive.
 *
 * A ping reloads and commits the volume, and a tab left open on Generate for a day would
 * do that seven hundred times to protect nothing — the sweep it drives also runs whenever
 * the list is read, which is the moment it matters.
 *
 * A tab that was hidden for longer than the grace period has to say it is back before
 * anything reads the list, or its own drafts look abandoned.
 */
export function keepAlive(on: boolean): () => void {
  if (!on) return () => undefined
  const t = window.setInterval(() => {
    if (!document.hidden) void beat()
  }, 120_000)
  const wake = () => {
    if (!document.hidden) void beat()
  }
  document.addEventListener('visibilitychange', wake)
  return () => {
    window.clearInterval(t)
    document.removeEventListener('visibilitychange', wake)
  }
}
