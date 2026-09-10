import { useState } from 'react'

import { Popover, usePopover } from '../ui/Popover'
import {
  applyTheme, castColor, CHROMA_MAX, compileMood, loadTheme, MIDNIGHT, POLAR,
  saveTheme, type Receipt, type VsnTheme,
} from './theme'

/**
 * Appearance, on a mark rather than a button.
 *
 * It is not under the gear. That sheet holds weights — checkpoints, LoRAs, the
 * GPU and the token that serve them — and every row in it costs a download or a
 * delete. Appearance costs nothing and answers to nobody, so a card in there
 * read as one more thing to decide before you could work.
 *
 * So the door is a 7px dot after the wordmark, and it is the state it opens:
 * untinted it is `--dim` and reads as punctuation, and a cast makes it that
 * cast. Nothing on the header announces the feature, and nothing has to be kept
 * in sync with it either. It is still a real button with a name — reachable by
 * tab, named by its tooltip — because hidden and quiet are different things.
 *
 * The panel keeps `form`, because `.form .frow` is the sampling popover's row
 * and this is the same kind of thing: values you set and watch, not a list you
 * pick from. It does not keep `menu`, which dresses every button inside it as a
 * full-width left-aligned row — right for a list of things you can do to one
 * object, and wrong for a switch, a Set and a Reset, which it stretched to the
 * width of the panel. `tint-`, not `cast-`: `.chip.cast` is the scene
 * composer's actors, and a bare `.cast` panel rule reached every one of them.
 *
 * Two poles, five named accents, three dials and a sentence. Hue and Chroma
 * make the accent; Warmth is the only one that reaches a surface, and it reaches
 * it on a paper↔slate axis rather than a free hue — that separation is the
 * design system's, not a detail of this panel. The sentence is the record and
 * the numbers are its receipt, the same relationship the prompt has with what
 * it compiles to.
 */

/* Named accents, at a chroma that is unmistakably a choice. They set hue and
   chroma only — never warmth — because that is the whole rule: the accent is
   free to be neon, the neutrals are not free to be anything but paper or slate.
   `Midnight` is not among them; that is Reset, and it says so. */
const ACCENTS: Array<{ label: string; hue: number; chroma: number }> = [
  { label: 'Ember', hue: 40, chroma: 0.17 },
  { label: 'Amber', hue: 78, chroma: 0.16 },
  { label: 'Meadow', hue: 135, chroma: 0.15 },
  { label: 'Glacier', hue: 220, chroma: 0.14 },
  { label: 'Ultraviolet', hue: 302, chroma: 0.2 },
]

function Slider({
  label, value, min, max, step, read, onChange, className,
}: {
  label: string
  value: number
  min: number
  max: number
  step: number
  read: string
  onChange: (v: number) => void
  className?: string
}) {
  return (
    <div className="frow">
      <span>{label}</span>
      <div className="tint-slide">
        <input type="range" className={className} min={min} max={max} step={step} value={value}
               aria-label={label} onChange={(e) => onChange(Number(e.target.value))} />
        <i className="tint-read">{read}</i>
      </div>
    </div>
  )
}

export function ThemeDot() {
  const [theme, setTheme] = useState<VsnTheme>(loadTheme)
  const [phrase, setPhrase] = useState(theme.phrase ?? '')
  const [receipts, setReceipts] = useState<Receipt[] | null>(null)
  const pop = usePopover()

  const commit = (t: VsnTheme) => {
    setTheme(t)
    applyTheme(t)
    saveTheme(t)
  }

  // A slider drag is not prose, so it drops the phrase: leaving it attached
  // would print a receipt for words that no longer describe the theme on screen.
  const nudge = (patch: Partial<VsnTheme>) => {
    setReceipts(null)
    commit({ ...theme, ...patch, phrase: undefined })
  }

  const compile = () => {
    if (!phrase.trim()) return
    const { theme: next, receipts: rec } = compileMood(phrase, theme)
    setReceipts(rec)
    commit(next)
  }

  const pole = (t: VsnTheme) => {
    setReceipts(null)
    setPhrase('')
    // Keeps the accent and the warmth across a pole change: the two poles are
    // one theme's two grounds, and dropping the colour on the way over made
    // Polar look like a different feature rather than the same one inverted.
    commit({ ...t, hue: theme.hue, chroma: theme.chroma, warmth: theme.warmth })
  }

  return (
    <>
      <button className="tint-dot" id="t-appearance" type="button" title="Appearance"
              aria-label="Appearance" onClick={pop.toggle}>
        <span />
      </button>

      {pop.open && (
        <Popover anchor={pop.anchor} className="form tint-panel" onClose={pop.close}>
          <div className="frow">
            <span>Ground</span>
            <div className="switch">
              <button type="button" className={theme.pole === 'midnight' ? 'on' : ''}
                      onClick={() => pole(MIDNIGHT)}>Midnight</button>
              <button type="button" className={theme.pole === 'polar' ? 'on' : ''}
                      onClick={() => pole(POLAR)}>Polar</button>
            </div>
          </div>

          <div className="frow">
            <span>Accent</span>
            <div className="tint-casts">
              {ACCENTS.map((c) => (
                <button key={c.label} type="button"
                        className={`tint-cast${theme.hue === c.hue && theme.chroma === c.chroma ? ' on' : ''}`}
                        style={{ ['--swatch' as string]: castColor({ ...c, pole: theme.pole, warmth: 0 }) }}
                        onClick={() => nudge({ hue: c.hue, chroma: c.chroma })}>
                  <span />{c.label}
                </button>
              ))}
            </div>
          </div>

          <Slider label="Hue" className="hue" value={theme.hue} min={0} max={360} step={1}
                  read={`${Math.round(theme.hue)}°`} onChange={(hue) => nudge({ hue })} />
          <Slider label="Chroma" value={theme.chroma} min={0} max={CHROMA_MAX} step={0.005}
                  read={theme.chroma === 0 ? 'mono' : theme.chroma.toFixed(3)}
                  onChange={(chroma) => nudge({ chroma })} />
          {/* The only dial that touches a surface, and it is deliberately not a
              hue: paper one way, slate the other, and nothing else reachable. */}
          <Slider label="Warmth" className="warmth" value={theme.warmth} min={-1} max={1} step={0.05}
                  read={theme.warmth === 0 ? 'neutral'
                    : `${theme.warmth > 0 ? 'paper' : 'slate'} ${Math.round(Math.abs(theme.warmth) * 100)}%`}
                  onChange={(warmth) => nudge({ warmth })} />

          <div className="frow">
            <span>Mood</span>
            <div className="row" style={{ gap: 6 }}>
              <input className="grow" placeholder="smoke and amber" autoComplete="off"
                     spellCheck={false} value={phrase} aria-label="Describe a mood"
                     onChange={(e) => setPhrase(e.target.value)}
                     onKeyDown={(e) => { if (e.key === 'Enter') compile() }} />
              <button className="s" type="button" onClick={compile}>Set</button>
            </div>
          </div>

          {/* A hint indented under a 74px label is a hint shaped like a value,
              so both of these span the row instead. */}
          {receipts && (
            <div className="frow">
              <i>{receipts.map((r) => `«${r.word}» ${r.effect}`).join(' · ')}</i>
            </div>
          )}

          <div className="tint-foot">
            <i>
              {theme.warmth === 0 && theme.chroma === 0 && theme.pole === 'midnight'
                ? 'Midnight writes nothing — the stylesheet is the theme.'
                : 'Remembered on this device.'}
            </i>
            <button className="t" type="button" onClick={() => {
              setReceipts(null)
              setPhrase('')
              commit(MIDNIGHT)
            }}>Reset</button>
          </div>
        </Popover>
      )}
    </>
  )
}
