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

/** Four, not six: the browser gives one origin about six connections, and the
 *  two held back are for what must never queue behind pictures — the status
 *  poll during a run, and a clip's range requests. */
const MAX_INFLIGHT = 4
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

/**
 * A plain <img> that retries a failed load — and nothing else.
 *
 * For the pictures that must not queue behind covers: the canvas still that
 * just finished rendering is the most important fetch on screen, so it keeps
 * the browser's own scheduling and priorities. What it must not keep is the
 * <img> element's silence on failure: one dropped response used to blank the
 * render somebody just waited minutes for, until a full reload.
 */
export function RetryImg({ src, ...rest }: React.ImgHTMLAttributes<HTMLImageElement>) {
  const tries = useRef(0)
  const [bust, setBust] = useState(0)
  const busted = bust && src
    ? `${src}${src.includes('?') ? '&' : '?'}r=${bust}`
    : src
  return (
    <img {...rest} src={busted}
         onError={() => {
           const wait = RETRY_MS[tries.current]
           if (wait == null) return
           tries.current += 1
           window.setTimeout(() => setBust(performance.now()), wait)
         }} />
  )
}

export function Thumb({ url, className, style, onClick }: {
  url: string
  className?: string
  /** The packed grid hands a card its aspect-ratio; passed through untouched. */
  style?: React.CSSProperties
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
    <img ref={el} className={className} style={style} alt="" src={src ?? undefined}
         onClick={onClick}
         title={failed
           ? 'The thumbnail could not be loaded after several tries — the full-size view may still work.'
           : undefined} />
  )
}
