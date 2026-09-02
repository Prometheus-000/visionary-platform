import { useEffect } from 'react'

import { Doors } from '../scene/Doors'
import { faceOf } from '../scene/model'
import { NumInput } from '../ui/NumInput'
import { shotGroup, shotItem, useStore } from '../store'
import { Glyph } from './Glyph'
import { shotLive } from './vocab'

/**
 * The pills you picked, in the order you picked them.
 *
 * A collapsed valued pill reads as what you chose rather than as a form; expanded,
 * it is an input and — for dialogue — a language select, because the guide names
 * the eleven and forbids inventing one.
 *
 * **It costs nothing while empty** — not a collapsed row, no element at all. That
 * rule was briefly reversed to put the palette's door at the head of this rail, and
 * the measurement put it back: a row that exists only to hold one button spends 34px
 * of a console capped at 30% of the viewport, at rest, forever. The doors ride the
 * field's trailing edge instead; see `Doors`. Once there is something on the rail
 * the row is carrying content, and then it is worth its height.
 *
 * **On the video side it is one rail with three kinds of chip** — cast, shot pills,
 * LoRAs. They were three mechanisms in two places and they are one tool: each is
 * something attached to the thing you are making, and each opens its own card when
 * touched. The image side keeps the pills alone, because `#lora-box` is a disclosure
 * under a prompt box and the video side no longer has one.
 */
export function Rail({ onPalette }: {
  onPalette: (e: React.MouseEvent<HTMLElement>) => void
}) {
  const s = useStore()
  const vocab = s.state?.shot_vocab ?? []
  const video = s.kind === 'video'

  // The pointer half of collapsing an expanded pill, and the load-bearing one.
  // focusout is the obvious mechanism and it only covers the case where focus
  // lands somewhere that takes it: click a canvas, a label, a dead area of the
  // bar, and focus does not move at all, so nothing fires and the pill stays a
  // form. Capture-phase mousedown is the pattern in this app that is known to
  // hold — it is what closes every popover.
  useEffect(() => {
    if (!s.shotOpen) return
    const down = (e: MouseEvent) => {
      if (!(e.target as Element | null)?.closest?.('.spill.open')) s.setShotOpen(null)
    }
    document.addEventListener('mousedown', down, true)
    return () => document.removeEventListener('mousedown', down, true)
  }, [s.shotOpen, s])

  return (
    <div className={`trail${video ? ' has-doors' : ''}`}>
      {/* Pinned, while the chips scroll past them. A door that scrolls away is a
          door you have to go looking for, which is most of what was wrong with
          the version of these that had no words. */}
      {video && <Doors onPalette={onPalette} />}
      <div className="wrap" id="shot-rail">
      {video && s.scene.cast.map((c) => {
        // **Their face, not a dot.** The chip carried a 5px circle, filled or
        // hollow, which was the one fact it could hold about somebody — and it is
        // the wrong fact and the wrong drawing: a person reduced to a monogram, in
        // a product whose strongest lever is that it can be handed a photograph.
        // The dot stays for a member with no picture, where it is still exactly
        // right: that name compiles to prose rather than to a `<Subject N>`.
        const face = faceOf(c, s.pool)
        return (
          <button key={c.id} type="button" data-cast={c.id}
                  className={`chip cast${s.railOpen === c.id ? ' on' : ''}`}
                  onClick={() => { s.setRailOpen(s.railOpen === c.id ? null : c.id) }}>
            {face
              ? <img className="av" src={face.url} alt="" draggable={false} />
              : <i className={`d${c.refs.length ? ' full' : ''}`} />}
            {c.name || c.kind}
          </button>
        )
      })}
      {s.shot.map((p) => {
        const g = shotGroup(vocab, p.key)
        const it = shotItem(vocab, p.key)
        if (!g || !it) return null
        const off = shotLive(s, g, it) ? '' : ' off'
        // mousedown, not click. An expanded pill's input has focus and its own
        // collapse rebuilds the rail, so by the time a click on ✕ would fire the
        // button it was aimed at has been replaced and the click lands on nothing.
        const remove = (e: React.MouseEvent) => {
          e.preventDefault()
          s.toggleShot(p.key)
        }

        if (it.valued && s.shotOpen === p.key) {
          return (
            <span key={p.key} className={`spill val open${off}`}>
              <Glyph cls={it.glyph} />
              {it.valued === 'dialogue' && (
                <select className="lang" value={p.lang ?? ''}
                        onChange={(e) => s.setPill(p.key, { lang: e.target.value })}>
                  {(s.state?.shot_langs ?? []).map((l) => <option key={l}>{l}</option>)}
                </select>
              )}
              <input autoComplete="off" className="v" autoFocus value={p.value ?? ''}
                     placeholder={it.hint || it.label}
                     onChange={(e) => s.setPill(p.key, { value: e.target.value })}
                     // Tabbing away collapses it too. Guarded on relatedTarget
                     // because the language select is inside the same pill, and a
                     // bare blur handler collapsed it the moment you reached for
                     // the one control the expansion exists to offer.
                     onBlur={(e) => {
                       if (!e.currentTarget.closest('.spill')?.contains(e.relatedTarget as Node))
                         s.setShotOpen(null)
                     }}
                     onFocus={(e) => e.currentTarget.setSelectionRange(
                       e.currentTarget.value.length, e.currentTarget.value.length)} />
              <button className="x" title="Remove" type="button" onMouseDown={remove} />
            </span>
          )
        }

        return (
          <span key={p.key} className={`spill${it.valued ? ' val' : ''}${off}`}
                onClick={(e) => {
                  if ((e.target as Element).closest('.x')) return
                  if (it.valued) s.setShotOpen(p.key)
                }}>
            <Glyph cls={it.glyph} />
            <b className={it.valued && p.value ? 'set' : ''}>
              {it.valued ? (p.value || it.label) : it.label}
              {/* A move that came off the storyboard with an amplitude or a
                  speed says so, in the same words the document will — the
                  guide omits medium and normal, and so does this. */}
              {p.amp && p.amp !== 'medium' ? ` · ${p.amp}` : ''}
              {p.speed && p.speed !== 'normal' ? ` · ${p.speed}` : ''}
            </b>
            <button className="x" title="Remove" type="button" onMouseDown={remove} />
          </span>
        )
      })}
      {video && s.loras.map((c) => (
        <span key={c.path} className="chip lora" data-lora={c.rel}>
          <span className="nm" title={c.path}>{c.rel}</span>
          {/* `fine` is 0.05 because the useful range is 1.0 to 1.4 — the same
              reason the numeric fields carry their own step. */}
          <NumInput className="val" value={String(c.strength)} fine={0.05} bigStep={0.25}
                    title="How hard this LoRA is applied."
                    onValue={(v) => {
                      const n = parseFloat(v)
                      if (Number.isFinite(n)) s.patchLora(c.path, { strength: n })
                    }} />
          <button type="button" className="x" title={`Remove ${c.rel}`}
                  onClick={() => { s.dropLora(c.path) }}>×</button>
        </span>
      ))}
      </div>
    </div>
  )
}
