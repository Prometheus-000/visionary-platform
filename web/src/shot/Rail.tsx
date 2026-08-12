import { useEffect } from 'react'

import { shotGroup, shotItem, useStore } from '../store'
import { Glyph } from './Glyph'
import { shotLive } from './vocab'

/**
 * The pills you picked, in the order you picked them.
 *
 * A collapsed valued pill reads as what you chose rather than as a form; expanded,
 * it is an input and — for dialogue — a language select, because the guide names
 * the eleven and forbids inventing one.
 */
export function Rail() {
  const s = useStore()
  const vocab = s.state?.shot_vocab ?? []

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
    <div className="wrap" id="shot-rail">
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
              <input className="v" autoFocus value={p.value ?? ''}
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
            </b>
            <button className="x" title="Remove" type="button" onMouseDown={remove} />
          </span>
        )
      })}
    </div>
  )
}
