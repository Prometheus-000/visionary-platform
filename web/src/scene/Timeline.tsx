import { useCallback, useRef, useState } from 'react'

import { useStore } from '../store'
import { SHOT_SECONDS, shares, type Shot } from './model'

/**
 * Time, along the axis time actually runs on.
 *
 * **Left to right, and authored.** The composer stacked shots vertically and
 * derived each one's length from how much you had written about it. Both halves
 * were wrong and the owner's objection retires them together:
 *
 * > Time is not derived from the description because doing so is impossible. If
 * > I'm a director, maybe I want a 5 minute scene where the protagonist sits in
 * > a chair. The director decides, not the model.
 *
 * Nothing about "he sits in the chair" implies five seconds or five minutes. So
 * a shot is a **bar you pull**, and the thing the derivation was protecting — a
 * 9px handle between two rows — is answered by giving duration its own axis
 * rather than by taking the decision away.
 *
 * **A fixed scale, and the track scrolls.** Fitting the whole scene to the width
 * would make every bar move while you drag one, and it would make a five-minute
 * scene unreachable — the two things this exists to allow. `PX_PER_SEC` is
 * constant, so a second is the same distance everywhere and the track is as long
 * as the film is.
 *
 * **The break is arithmetic, not a setting.** H3 tops out around 14.4s, so a
 * scene longer than one generation is several, and the mark falls wherever the
 * running total crosses the cap. See `Continue` on the canvas: what carries
 * across a break is the cast, the look and the last frame, which is context
 * rather than a capability.
 */

/** One second, in pixels. A beat wants to be visible at a glance and a bar
 *  wants an edge big enough to grab, which 30px gives at the 1s floor. */
const PX_PER_SEC = 30
/** A generation's ceiling — `H3_MAX_FRAMES` / `H3_FPS`, 345/24. A single shot
 *  cannot outrun one take, because there is no cut inside a shot to break it at. */
const TAKE_MAX = 14
const SHOT_MIN = 1

const tick = (t: number) => {
  const m = Math.floor(t / 60)
  const s = Math.round(t - m * 60)
  return m ? `${String(m)}:${String(s).padStart(2, '0')}` : `${String(s)}s`
}

export function Timeline() {
  const s = useStore()
  const shots = s.scene.shots
  const secs = shares(shots)
  const total = secs.reduce((n, x) => n + x, 0)
  const track = useRef<HTMLDivElement>(null)
  /** Which bar is being pulled, so the whole track can suppress selection and
   *  the cursor can stay `col-resize` past the 4px edge the pointer left. */
  const [pulling, setPulling] = useState<string | null>(null)

  const pull = useCallback((shot: Shot, startX: number, from: number) => {
    setPulling(shot.id)
    const move = (e: PointerEvent) => {
      const next = Math.round(
        Math.min(TAKE_MAX, Math.max(SHOT_MIN, from + (e.clientX - startX) / PX_PER_SEC)) * 2) / 2
      useStore.getState().patchShot(shot.id, { beats: next })
    }
    const up = () => {
      setPulling(null)
      window.removeEventListener('pointermove', move)
      window.removeEventListener('pointerup', up)
    }
    window.addEventListener('pointermove', move)
    window.addEventListener('pointerup', up)
  }, [])

  // Where each take ends, by the running total crossing the cap. Computed here
  // rather than stored: it is a consequence of the bars, and a stored copy is a
  // second thing to keep in step with them.
  const breaks = new Set<number>()
  let run = 0
  secs.forEach((sec, i) => {
    if (run + sec > TAKE_MAX) { breaks.add(i - 1); run = sec } else { run += sec }
  })

  let at = 0
  return (
    <div className={`tl${pulling ? ' pulling' : ''}`}>
      <div className="tl-track" ref={track}>
        {shots.map((shot, i) => {
          const sec = secs[i]!
          const start = at
          at += sec
          const sel = shot.id === s.shotSel
          return (
            <div key={shot.id} className={`tl-shot${sel ? ' sel' : ''}`}
                 style={{ width: sec * PX_PER_SEC }}
                 onPointerDown={(e) => {
                   if ((e.target as HTMLElement).closest('.tl-pull')) return
                   s.selectShot(shot.id)
                   // Focus follows selection, because the bar and the field under
                   // it are one control: picking a shot is picking what to write.
                   requestAnimationFrame(() => document.getElementById('prompt')?.focus())
                 }}>
              <span className="tl-n">{i + 1}</span>
              {/* The prompt, in the bar. Not a thumbnail — there is no frame yet,
                  and the sentence is the only thing that says what this shot is. */}
              <span className="tl-line">{shot.line.trim() || <i>…</i>}</span>
              <span className="tl-secs">{sec % 1 ? sec.toFixed(1) : sec}s</span>
              {/* The whole right edge, not a hairline: this is the one control
                  the old derivation existed to avoid, so it has to be grabbable. */}
              <span className="tl-pull" title={`${tick(start)} → ${tick(start + sec)} — drag to set how long this shot runs`}
                    onPointerDown={(e) => { e.preventDefault(); e.stopPropagation(); pull(shot, e.clientX, sec) }} />
              {breaks.has(i) && <span className="tl-cut" title="H3 renders about 14 seconds at a time — the next take picks up from this frame" />}
            </div>
          )
        })}
        <button type="button" className="tl-add" title="Another shot"
                onClick={() => {
                  const id = s.addShot(shots[shots.length - 1]?.id)
                  requestAnimationFrame(() => document.getElementById('prompt')?.focus())
                  return id
                }}>+</button>
      </div>
      {/* A ruler under the bars, at whole seconds while they are far apart and
          every five once they are not. It is a readout: nothing is dragged here. */}
      <div className="tl-rule" style={{ width: total * PX_PER_SEC }}>
        {Array.from({ length: Math.floor(total) + 1 }, (_, n) => n)
          .filter((n) => (total > 40 ? n % 10 === 0 : total > 16 ? n % 5 === 0 : n % 2 === 0))
          .map((n) => (
            <span key={n} className="tl-tick" style={{ left: n * PX_PER_SEC }}>{tick(n)}</span>
          ))}
      </div>
    </div>
  )
}

export { PX_PER_SEC, TAKE_MAX, SHOT_SECONDS }
