/**
 * The last frame of a clip, as the base64 a keyframe slot takes.
 *
 * **This is the whole of what chaining needs from the model, and the model does
 * not need to do it.** A scene longer than one generation is several generations
 * — H3 tops out at `H3_MAX_FRAMES`, 345 at 24fps, so about 14.4 seconds — and the
 * thing that makes them read as one continuous scene rather than as unrelated
 * clips is that the next one opens where the last one stopped. `first_frame`
 * already exists on `/api/video` and already promotes the task to `i2va`; nothing
 * was ever handing it a frame.
 *
 * Client-side on purpose. The bytes are already on the volume and already served
 * to a `<video>` on the canvas, so a route that decoded a frame server-side would
 * be a GPU-adjacent container doing work a decoder in the page does for free —
 * which is the "never rent a GPU to do CPU work" rule one step further out.
 *
 * `seekable.end` rather than `duration`: a fragmented MP4 can report `Infinity`
 * for the latter before the whole file is buffered, and seeking to `Infinity`
 * resolves to nothing and hangs the promise. A sixteenth of a second back from
 * the end, because seeking exactly to it lands past the final sample on some
 * decoders and paints black.
 */
const BACK_OFF = 1 / 16

export function lastFrame(src: string): Promise<string | null> {
  return new Promise((resolve) => {
    const v = document.createElement('video')
    // Same origin — the file route serves it — so the canvas stays untainted and
    // `toDataURL` is allowed. Without this it throws a SecurityError instead.
    v.crossOrigin = 'anonymous'
    v.muted = true
    v.preload = 'auto'

    // Nothing here can be allowed to hang the caller: a codec the browser will
    // not decode simply means no keyframe, and the next take opens cold.
    const bail = window.setTimeout(() => { done(null) }, 8000)
    let settled = false
    const done = (out: string | null) => {
      if (settled) return
      settled = true
      window.clearTimeout(bail)
      v.removeAttribute('src')
      v.load()
      resolve(out)
    }

    v.onerror = () => { done(null) }
    v.onloadeddata = () => {
      const end = v.seekable.length ? v.seekable.end(v.seekable.length - 1) : v.duration
      if (!Number.isFinite(end) || end <= 0) { done(null); return }
      v.currentTime = Math.max(0, end - BACK_OFF)
    }
    // `seeked` rather than `timeupdate`: the frame is only painted once the seek
    // has actually completed, and drawing early gives you whatever was decoded
    // last — in practice frame zero.
    v.onseeked = () => {
      try {
        const c = document.createElement('canvas')
        c.width = v.videoWidth
        c.height = v.videoHeight
        if (!c.width || !c.height) { done(null); return }
        c.getContext('2d')?.drawImage(v, 0, 0)
        // The same shape the pool and the keyframe slots hold: a bare base64
        // payload with no data-URI prefix, which is what `/api/video` takes.
        done(c.toDataURL('image/jpeg', 0.92).split(',')[1] ?? null)
      } catch {
        done(null)
      }
    }
    v.src = src
  })
}
