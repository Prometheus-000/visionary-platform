import { useEffect } from 'react'

import { usePopover } from '../ui/Popover'
import { shotGroup, shotItem, useStore } from '../store'
import { Glyph } from './Glyph'
import { Palette } from './Palette'
import { shotLive } from './vocab'

/**
 * The door to the vocabulary, and the pills you picked, in the order you picked them.
 *
 * A collapsed valued pill reads as what you chose rather than as a form; expanded,
 * it is an input and — for dialogue — a language select, because the guide names
 * the eleven and forbids inventing one.
 *
 * **The door is here rather than in the settings strip, and that is the whole point
 * of this arrangement.** It was a 34px glyph between the size button and `+ LoRA`,
 * which is a row of controls you reach for when you already know what you want —
 * so the one feature that has to *announce itself* was filed where nobody looks
 * for a capability, next to a second opaque glyph doing the same thing for regions.
 * Its home is the prompt: these are words that go into the sentence, and the rail
 * they land in is directly under the field. A door at the head of the thing it
 * fills says what it is without a caption.
 *
 * It is a sibling of the scrolling rail rather than its first child, so sixteen
 * pills scrolling right cannot carry the way back to the palette off the edge.
 */
export function Rail() {
  const s = useStore()
  const vocab = s.state?.shot_vocab ?? []
  const pal = usePopover()

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
    <div className="rail-row">
      {/* Words, not a bare glyph, and the rule that governs the strip is what says
          so: an icon can carry a control whose home you are already in, and it
          cannot announce a *destination* to someone who does not know it exists.
          This is a destination — eighty-seven tiles behind one press.

          The tile is the teaser, and it is the one place in this app where an icon
          can teach: a dolly-out is a thing neither the word nor a static picture
          shows you. It animates on hover and is frozen otherwise, because the page
          at rest runs nothing — the same reason the pills' own copies are paused.
          The glyph follows the kind, so the tease is never a move the thing in
          front of you cannot make: the camera group is video-only, and on the
          image side a pulled-out framing is what a shot size means there. */}
      <button type="button" id="shot-add" onClick={pal.toggle}
              className={`shot-add${s.shot.length ? '' : ' hero'}`}
              title="Framing, angle, light and tone — the words this model was trained on.">
        {s.shot.length
          ? <b>+ Shot</b>
          : <><Glyph cls={s.kind === 'video' ? 'ca-pull' : 'fr-mcu'} /><b>Shot</b></>}
      </button>

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

      {/* One palette, and it lives with the one door that opens it. It used to be
          rendered by the console so that the image and video glyphs could share it;
          there is one trigger now, so the popover belongs to it. */}
      {pal.open && <Palette anchor={pal.anchor} onClose={pal.close} />}
    </div>
  )
}
