import { useLayoutEffect, useRef } from 'react'

import { caretProps } from '../lora/caret'
import { moveClause } from '../console/moveClause'
import { growRows } from '../console/fieldMax'
import { resolveVid } from '../console/resolve'
import { supports, useStore } from '../store'
import { handleOf, shares, times, type Shot } from './model'

/**
 * The timeline, where the video side's prompt box used to be.
 *
 * **One shot is the prompt box.** Same id, same placeholder, same mirror, same
 * chords — because that is the degrade this whole layer rests on: with one shot
 * and nothing else chosen the compiler returns the typed text byte-for-byte, and
 * a surface that looked different while doing exactly the same thing would be
 * announcing a feature it is not yet using.
 *
 * A second shot is what makes it a timeline, and **the rows divide the field's
 * existing allowance rather than adding to it** — one prompt at two lines and
 * two shots at one line each are the same height, so a four-shot scene costs
 * what a long prompt costs. See `growRows`.
 */
export function Shots({ consoleEl, hide, onSubmit }: {
  consoleEl: React.RefObject<HTMLDivElement | null>
  /** The negative box is showing instead. Hidden rather than unmounted, so the
   *  caret, the scroll position and the selection are where you left them when
   *  you switch back — the same reason `#prompt` was only ever `.hide`d. */
  hide: boolean
  onSubmit: () => void
}) {
  const s = useStore()
  const box = useRef<HTMLDivElement>(null)
  const shots = s.scene.shots
  const many = shots.length > 1
  // A string on the composer, because an empty box means "the model's default"
  // — see `ResolvedVid`. The clock needs a number, and the fallback is one
  // second so a model with no length yet still divides rather than dividing by
  // zero; nothing is shown at that point anyway.
  const secs = Number(resolveVid(s).seconds) || 1
  const cuts = times(shots, secs)

  useLayoutEffect(() => {
    growRows(box.current, consoleEl.current)
  })

  return (
    <div className={`tline${many ? ' many' : ''}${hide ? ' hide' : ''}`} ref={box}>
      {/* A readout, not an input: a shot's share of the clip is the length of
          what you wrote about it, so there is nothing to drag and nothing to
          miss with a thumb. It only appears once there is a division to show —
          a single full-width bar is a picture of the fact that there are no
          cuts. */}
      {many && <Strip />}
      {shots.map((shot, i) => (
        <Row key={shot.id} shot={shot} n={i} at={cuts[i]?.[0] ?? 0}
             many={many} onSubmit={onSubmit} />
      ))}
    </div>
  )
}

function Strip() {
  const s = useStore()
  const w = shares(s.scene.shots)
  return (
    <div className="tstrip">
      {s.scene.shots.map((shot, i) => (
        <i key={shot.id} style={{ flex: w[i] }}
           className={shot.id === s.shotSel ? 'sel' : undefined} />
      ))}
    </div>
  )
}

/** `MM:SS.mmm` is the cut format the document takes; the gutter is a readout for
 *  a person, so it is the same instant at the precision a person reads. */
const tick = (t: number) => {
  const m = Math.floor(t / 60)
  const rest = (t - m * 60).toFixed(2).padStart(5, '0')
  return m ? `${String(m)}:${rest}` : rest
}

function Row({ shot, n, at, many, onSubmit }: {
  shot: Shot; n: number; at: number; many: boolean; onSubmit: () => void
}) {
  const s = useStore()
  const area = useRef<HTMLTextAreaElement>(null)
  const mirror = useRef<HTMLDivElement>(null)
  // The first row keeps `#prompt`. Everything that reaches for the prompt by id
  // — the stray-key focus in App.tsx, the checks, the Enter binding — is asking
  // for "the box you write in", and that is still this one.
  const id = n === 0 ? 'prompt' : `shot-${shot.id}`
  const write = (line: string) => { s.patchShot(shot.id, { line }) }

  const hint = n > 0
    ? 'What happens next…'
    : supports(s).audio
      ? 'Describe the shot, the motion — and the audio: dialogue, effects, music…'
      : 'Describe the shot and the motion…'

  const keys = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    const el = e.currentTarget
    if (e.altKey && !e.metaKey && !e.ctrlKey && !e.nativeEvent.isComposing
        && (e.key === 'ArrowLeft' || e.key === 'ArrowRight')) {
      const moved = moveClause(shot.line, el.selectionStart ?? 0,
                               e.key === 'ArrowRight' ? 1 : -1)
      if (!moved) return
      e.preventDefault()
      write(moved.value)
      requestAnimationFrame(() => { el.setSelectionRange(moved.caret, moved.caret) })
      return
    }
    // ⌫ on an empty row past the first removes it, which is the only way back
    // out of a timeline that does not need a control of its own. The first row
    // is never removable — a scene with no shots is a scene with nowhere to
    // type, and `_validate_scene` reads no shots as no scene at all.
    if (e.key === 'Backspace' && many && n > 0 && !shot.line && !shot.say.text) {
      e.preventDefault()
      s.dropShot(shot.id)
      return
    }
    if (e.key === 'Enter' && !e.nativeEvent.isComposing && !e.shiftKey && !e.altKey) {
      e.preventDefault()
      // ⌘⏎ submits from anywhere; a bare ⏎ at the end of a row that already has
      // something in it starts the next shot, which is the gesture a timeline
      // makes available and a prompt box cannot.
      if (e.metaKey || e.ctrlKey || !shot.line.trim()) { onSubmit(); return }
      const next = s.addShot(shot.id)
      requestAnimationFrame(() => document.getElementById(`shot-${next}`)?.focus())
    }
  }

  return (
    <div className={`trow${shot.id === s.shotSel ? ' sel' : ''}`}>
      {many && (
        <span className="tnum" aria-hidden="true">
          {n + 1}<em>{tick(at)}</em>
        </span>
      )}
      <div className="tbox">
        <div className="mk-mirror" ref={mirror} aria-hidden="true">
          <Painted line={shot.line} />
          {/* Load-bearing empty span — a mirror that ends exactly at its last
              character loses the newline just typed, and the copy behind the box
              stops matching the box by one line. */}
          <span />
        </div>
        <textarea id={id} ref={area} rows={1} placeholder={hint} value={shot.line}
                  onScroll={(e) => {
                    if (mirror.current) mirror.current.scrollTop = e.currentTarget.scrollTop
                  }}
                  onChange={(e) => { write(e.target.value) }}
                  onKeyDown={keys}
                  // Spread first, then extend the one event this row also needs.
                  // `caretProps` carries its own `onFocus` and a later prop wins,
                  // so declaring one above it would silently drop the caret sink
                  // and take ⌥←/→ with it.
                  {...caretProps('prompt', write)}
                  onFocus={(e) => {
                    caretProps('prompt', write).onFocus(e)
                    s.selectShot(shot.id)
                  }} />
      </div>
    </div>
  )
}

/**
 * The line with its mentions marked.
 *
 * A mention is stored as the literal text `@ava` and *painted* as a chip, which
 * is the whole reason the mirror survived the deletion of the marks it was built
 * for. The consequence is worth stating rather than discovering: edit the handle
 * and it stops being a mention — the words turn plain and the shot no longer
 * claims that subject.
 *
 * A handle nobody defined is marked as missing rather than left plain, because
 * the failure it prevents reads as the model ignoring you: `@ava` compiles to
 * those literal characters, which the encoder renders as nothing at all.
 */
function Painted({ line }: { line: string }) {
  const cast = useStore((st) => st.scene.cast)
  const known = new Set(cast.map((c) => handleOf(c.name)).filter(Boolean))
  const out: React.ReactNode[] = []
  const re = /@([a-z0-9_]+)/gi
  let at = 0
  for (const m of line.matchAll(re)) {
    const i = m.index
    if (i > at) out.push(line.slice(at, i))
    out.push(
      <span key={i} className={`men${known.has(m[1]!.toLowerCase()) ? '' : ' miss'}`}>
        {m[0]}
      </span>,
    )
    at = i + m[0].length
  }
  out.push(line.slice(at))
  return <>{out}</>
}
