/**
 * The one seam between the page and the 33 routes.
 *
 * Ported from `api()`/`post()` in UI_HTML rather than replaced, because the
 * contract is load-bearing and unobvious: **a failed request comes back as
 * `{error}`, it never throws.** A 500 from Modal is a plain-text traceback, so
 * `r.json()` rejects inside whichever caller happened to make the request and
 * takes the rest of that function with it — `loadState()` died on
 * `s.models.filter` and left the composer looking like a deployment with no
 * weights on it. Every caller here is written expecting to test for `error`,
 * so a `fetch` wrapper that throws would break them all silently.
 *
 * The typing follows from that: `Res<T>` is `T | ApiError`, and `failed()` is
 * the narrowing guard. There is deliberately no `unwrap()` that throws — it
 * would reintroduce exactly the failure this shape exists to prevent.
 */

/**
 * `error` is the sentence a person reads; `detail` is what the server actually said.
 *
 * They were one field, and the field was
 * `${status} ${statusText} ${body.slice(0, 400)}` — so the headline of every failure on
 * this page was 400 characters of Modal traceback, in an `alert()` or an err-box, with
 * the one useful word somewhere in the middle of it. A person cannot act on that, and
 * the person who *can* act on it wants the whole thing rather than the first 400 bytes.
 * Splitting it serves both: the sentence says what to do, `detail` rides along for the
 * disclosure to open, and nothing is thrown away.
 */
export type ApiError = { error: string; detail?: string }
export type Res<T> = T | ApiError

/** Narrowing guard. `if (failed(r)) return r.error` is the whole idiom. */
export function failed<T>(r: Res<T>): r is ApiError {
  return !!r && typeof r === 'object' && 'error' in r && typeof (r as ApiError).error === 'string'
}

async function api<T>(path: string, init?: RequestInit): Promise<Res<T>> {
  let r: Response
  try {
    r = await fetch(path, init)
  } catch (e) {
    // `Failed to fetch` is what the browser says for a server that is not running, a
    // dev proxy pointed at the wrong port, and a dropped wifi connection alike. The
    // page cannot tell those apart, so it says the one thing true of all three and
    // keeps the browser's wording underneath for whoever can.
    return {
      error: 'Could not reach the server — check that it is running, then try again.',
      detail: String(e instanceof Error ? e.message : e),
    }
  }
  const body = await r.text()
  try {
    return JSON.parse(body) as T
  } catch {
    return { error: statusNote(r.status, r.statusText), detail: body.trim() || undefined }
  }
}

/**
 * A status, as a next step rather than a number.
 *
 * Only reached when the body would not parse, which on this stack means the server did
 * not get as far as writing JSON: a traceback, a proxy's HTML error page, or an empty
 * 502 from a container that has not finished starting. The number stays in the sentence
 * because it is what you quote when you ask someone; it is no longer the whole message.
 */
function statusNote(status: number, statusText: string): string {
  if (status === 404) {
    return `Not found (404) — this build is asking for a route the server does not have,`
      + ` which usually means the page and the deployment are different versions.`
  }
  if (status === 401 || status === 403) {
    return `Not authorised (${status}) — check the HuggingFace token under Settings.`
  }
  if (status === 502 || status === 503 || status === 504) {
    return `The server did not answer (${status}) — it may still be starting up,`
      + ` so give it a moment and try again.`
  }
  if (status >= 500) {
    return `The server hit an error (${status}) partway through this request; its own`
      + ` report is below.`
  }
  return `Request failed (${status}${statusText ? ` ${statusText}` : ''}).`
}

function post<T>(path: string, body?: unknown): Promise<Res<T>> {
  return api<T>(path, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body ?? {}),
  })
}

/**
 * `setInterval` for a poll, with the one thing `setInterval` cannot do: skip a
 * tick while the last one is still out. Ported verbatim, and deliberately NOT
 * replaced with `useEffect` + `setInterval`, which reintroduces the overlap.
 *
 * Every poll here awaits a request, and `setInterval` fires on a clock rather
 * than on a reply. `/api/status` reads a *network* Dict, so at 400ms a slow
 * reply does not delay the next tick, it overlaps it, and three things follow.
 *
 * Responses land out of order, so the bar is painted by whichever reply
 * arrives last rather than whichever is newest: step 14 lands, then step 12
 * lands on top of it and the bar walks backwards. That is the client half of
 * "the progress bar goes nuts"; `_publish`'s lock is the server half.
 *
 * And in-flight polls hold connections. A browser gives one origin about six,
 * and everything on this page comes off that origin — the gallery's covers,
 * the stills on the canvas, and a `<video>` re-requesting byte ranges for as
 * long as it plays. A pile of polls starves them: the clip stutters and the
 * grid comes back half-painted, neither of which looks like a poll loop.
 */
export function everyMs(fn: () => Promise<unknown>, ms: number): number {
  let busy = false
  return window.setInterval(async () => {
    if (busy) return
    busy = true
    try {
      await fn()
    } finally {
      busy = false
    }
  }, ms)
}

export { api, post }
