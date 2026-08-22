import type { CSSProperties } from 'react'
import type { ApiError } from '../api/client'

/**
 * The err-box, plus somewhere for the traceback to live.
 *
 * `.err-box` was `<div className="err-box">{someString}</div>` at eight call sites, and
 * the string arrived from `api()` as status + statusText + 400 characters of whatever
 * the server printed. So the loudest thing on screen was a stack trace, and the sentence
 * a person could act on was not there at all.
 *
 * `client.ts` now splits those: `error` is the sentence, `detail` is the server's own
 * report. This renders the first and folds the second into a `<details>` — closed, so it
 * costs a line of chrome and no attention, open for the one reader in ten who wants it.
 * Deleting the traceback instead would have been the easy version of this fix and the
 * wrong one: it is the only thing in the failure that says what actually broke.
 *
 * Takes a bare string too, because several of the call sites raise their own messages
 * that never had a server behind them ("Give the set a name to save it.") and those
 * should not have to be dressed up as an ApiError to render.
 */
export function ErrorNote({ err, style }: {
  err: string | ApiError | null | undefined
  style?: CSSProperties
}) {
  if (!err) return null
  const message = typeof err === 'string' ? err : err.error
  const detail = typeof err === 'string' ? undefined : err.detail
  if (!message && !detail) return null

  return (
    <div className="err-box" style={style}>
      {message}
      {detail && (
        <details className="err-detail">
          <summary>What the server said</summary>
          <pre>{detail}</pre>
        </details>
      )}
    </div>
  )
}
