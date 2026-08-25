import { useEffect, useRef, useState } from 'react'

import { useNearViewport } from './inview'

/**
 * A dataset thumbnail that waits its turn and does not give up.
 *
 * A bare `<img loading="lazy">` was the first version, and it failed twice
 * over on a big set. The web container takes 20 requests at a time and the
 * browser multiplexes every visible tile onto one HTTP/2 connection, so an
 * 80-image drop fired its whole grid of thumbnail requests at once — the burst
 * queued out the listing and the heartbeat behind it, and the sheet read as
 * dead. And an `<img>` whose request failed (a container restart, a 500, a
 * dropped connection) shows an empty frame *forever*: the element never
 * retries, so a transient fault became a permanently blank tile that only a
 * full reload could repaint.
 *
 * So: at most `MAX_INFLIGHT` thumbnail fetches at once, in mount order, gated
 * to near the viewport, and a failure retries with backoff before it is
 * allowed to stay blank. Successes are kept as object URLs for the life of the
 * page, so a filter flip or a density change never refetches.
 */

const MAX_INFLIGHT = 6
const RETRY_MS = [800, 2500, 6000]

let inflight = 0
const waiters: (() => void)[] = []

const acquire = () =>
  new Promise<void>((res) => {
    if (inflight < MAX_INFLIGHT) {
      inflight += 1
      res()
    } else {
      waiters.push(() => {
        inflight += 1
        res()
      })
    }
  })

const release = () => {
  inflight -= 1
  waiters.shift()?.()
}

/** url → object URL, never revoked: the whole point is that a tile that loaded
 *  once is a tile that stays loaded. A few hundred 320px JPEGs is megabytes. */
const loaded = new Map<string, string>()

export function Thumb({ url, className, onClick }: {
  url: string
  className?: string
  onClick?: () => void
}) {
  const el = useRef<HTMLImageElement>(null)
  const near = useNearViewport(el, true)
  const [src, setSrc] = useState<string | null>(() => loaded.get(url) ?? null)
  const [failed, setFailed] = useState(false)

  useEffect(() => {
    if (!near || src || failed) return
    let alive = true
    void (async () => {
      for (let attempt = 0; ; attempt++) {
        await acquire()
        try {
          if (!alive) return
          const r = await fetch(url)
          if (r.ok) {
            const u = URL.createObjectURL(await r.blob())
            loaded.set(url, u)
            if (alive) setSrc(u)
            return
          }
        } catch {
          /* a network fault retries the same way a bad status does */
        } finally {
          release()
        }
        if (!alive) return
        const wait = RETRY_MS[attempt]
        if (wait == null) {
          setFailed(true)
          return
        }
        await new Promise((res) => setTimeout(res, wait))
        if (!alive) return
      }
    })()
    return () => {
      alive = false
    }
  }, [near, src, failed, url])

  return (
    <img ref={el} className={className} alt="" src={src ?? undefined}
         onClick={onClick}
         title={failed
           ? 'The thumbnail could not be loaded after several tries — the full-size view may still work.'
           : undefined} />
  )
}
