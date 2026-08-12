import { useEffect, useRef } from 'react'

import { IconArrange, IconOutfit, IconPhoto, IconScene, IconTrash } from '../icons'
import { Menu } from '../ui/Menu'
import { NumInput } from '../ui/NumInput'
import { usePopover } from '../ui/Popover'
import { DropTile } from '../media/DropTile'
import { shrinkB64 } from '../media/files'
import { caretProps, dropCaret } from '../lora/caret'
import { NEED_EDIT_LORA } from '../lora/note'
import { loraIndex, setTokenStrength, tokenStrength } from '../lora/tokens'
import { useStore } from '../store'
import { clamp01, distribute } from './geometry'

const PLATE_TITLE = {
  scene: 'Scene photo. The picture is generated inside it — lighting, perspective '
    + 'and shadows integrate.',
  outfit: 'Outfit or object photo. Transferred onto the subjects rather than pasted '
    + 'into the frame.',
} as const

/**
 * One row for whichever box is selected, not one row per box.
 *
 * What you touch on the canvas decides what this is about, which is why it can stay a
 * single 36px line at eight regions where the old stack was ~592px — the feature's
 * fullest state broke the rule the console exists to hold. The four coordinates are
 * still here: they are the escape hatch, and while you drag they are the readout that
 * teaches what they mean.
 */
export function RegionBar({ onReveal, Map: MapCmp }: {
  onReveal: () => void
  Map: React.ComponentType<{ onReveal: () => void }>
}) {
  const s = useStore()
  const field = useRef<HTMLInputElement>(null)
  const arrange = usePopover()
  const r = s.regions[s.rsel]
  const index = loraIndex(s.state)
  const strength = r ? tokenStrength(index, r.prompt) : null

  useEffect(() => {
    const el = field.current
    return () => { if (el) dropCaret(el) }
  }, [])

  if (!(s.regional && s.kind === 'image' && r)) return null

  const num = (key: 'x' | 'y' | 'w' | 'h', label: string, title: string) => (
    <div className="opt n" data-lb={label}>
      <span className="lead">{label}</span>
      <NumInput value={fmt(r[key])} fine={0.01} bigStep={0.1} title={title}
                onValue={(v) => {
                  const n = parseFloat(v)
                  if (Number.isFinite(n)) s.patchRegion(s.rsel, { [key]: clamp01(n) })
                }} />
    </div>
  )

  return (
    <div id="region-bar" className="opts">
      <MapCmp onReveal={onReveal} />

      <DropTile id="r-ref" label="Photo" value={r.ref} glyph={<IconPhoto />}
                title="A photo of this character. Pulls the box toward that likeness during sampling — stacks with the LoRA, and works without one."
                onFile={async (f) => {
                  const b64 = await shrinkB64(f)
                  if (b64) s.patchRegion(s.rsel, { ref: b64 })
                }}
                onClear={() => s.patchRegion(s.rsel, { ref: null })} />

      {/* Direction for one performer, not a second scene description. The split is
          the only rule this feature has and it is worth the two sentences: the prompt
          above is what every performer is standing in, this is what this one is
          doing, and the box already said where.

          A region's own field takes the same `<lora:…>` syntax as the main prompt,
          and that is the whole reason a region has no LoRA select. The two fields
          mean one thing at two scopes — a token in the main prompt is the canvas, a
          token in a box is that box — so nothing has to explain which is which. */}
      <div className="opt wide">
        <input id="r-prompt" ref={field} value={r.prompt}
               placeholder="a man in a denim jacket, laughing <lora:name:1.3>"
               title="This performer only — who they are and what they are doing. The scene, the light and the lens go in the prompt above; where they stand is the box. Do not write a position here, it is already said."
               onChange={(e) => s.patchRegion(s.rsel, { prompt: e.target.value })}
               {...caretProps('region', (v) => useStore.getState().patchRegion(
                 useStore.getState().rsel, { prompt: v }))} />
      </div>

      {/* A *view* of the first number in the box's token, not a second place the
          number lives. Blank and disabled when there is no token to read, because a
          strength with nothing to apply to is a number that lies. */}
      <div className="opt n" data-lb="Strength">
        <span className="lead">Strength</span>
        <NumInput id="r-strength" value={strength === null ? '' : String(strength)}
                  disabled={strength === null} placeholder={strength === null ? '—' : ''}
                  fine={0.05} bigStep={0.25}
                  title="How hard this box's LoRA is applied. The node pack's guidance is 1.3–1.4 for a character. Writes the number in the token."
                  onValue={(v) => {
                    const n = parseFloat(v)
                    if (Number.isFinite(n))
                      s.patchRegion(s.rsel, { prompt: setTokenStrength(index, r.prompt, n) })
                  }} />
      </div>

      {num('x', 'X', 'Left edge, as a fraction of the width. 0 is the left of the canvas.')}
      {num('y', 'Y', 'Top edge, as a fraction of the height. 0 is the top of the canvas.')}
      {num('w', 'W', 'How much of the canvas width this box covers, 0 to 1.')}
      {num('h', 'H', 'How much of the canvas height this box covers, 0 to 1.')}

      {/* Render-scoped, unlike everything to the left of the rule, which is about
          whichever box is selected. They lived in Advanced, which meant the two
          controls that only exist when regions are armed were behind a drawer that
          had nothing else to do with regions. */}
      <button className="opt ib" id="g-arrange" title="Distribute the boxes evenly"
              type="button" onClick={arrange.toggle}>
        <IconArrange />
      </button>
      {arrange.open && (
        <Menu anchor={arrange.anchor} onClose={arrange.close} items={[
          { label: 'Distribute in columns', run: () => s.setRegions(distribute(s.regions, true)) },
          { label: 'Distribute in rows', run: () => s.setRegions(distribute(s.regions, false)) },
        ]} />
      )}

      <div className="opt n" id="g-region-base-wrap" data-lb="Global">
        <span className="lead">Global</span>
        <NumInput id="g-region-base" value={s.regionWeight} fine={0.05} bigStep={0.25}
                  title="Multiplies every region's LoRA strength at once. 1 uses the strengths as written."
                  onValue={s.setRegionWeight} />
      </div>

      {/* The plates ride on two conditions, not one: regions on, and the weight they
          need actually downloaded. The two get different treatments, and the split is
          the point. Regions off: the tiles are gone, because there is nothing to put
          a scene behind. Weight missing: the tiles are *locked and dimmed*, because
          an install without the edit LoRA is one download away from scene and outfit
          transfer and had no way of learning either existed. A model-gated control
          the model will never read stays hidden; a weight-gated control is not that —
          it is a purchase you have not made yet, and hiding it hides the decision
          rather than the capability.

          Region photos ride on neither: a mold is not an extra_ref plate, so it needs
          no edit LoRA and never switches paths. */}
      {(['scene', 'outfit'] as const).map((slot) => (
        <DropTile key={slot} id={`g-drop-${slot}`} label={slot === 'scene' ? 'Scene' : 'Outfit'}
                  value={s.plate[slot]} locked={!s.state?.edit_lora}
                  glyph={slot === 'scene' ? <IconScene /> : <IconOutfit />}
                  title={s.state?.edit_lora ? PLATE_TITLE[slot] : NEED_EDIT_LORA}
                  onFile={async (f) => {
                    const b64 = await shrinkB64(f)
                    if (b64) s.setPlate(slot, b64)
                  }}
                  onClear={() => s.setPlate(slot, null)} />
      ))}

      <span className="actions">
        <button className="opt ib" id="r-del" type="button"
                title="Remove this box — or select it and press ⌫"
                onClick={() => {
                  s.setRegions(s.regions.filter((_, n) => n !== s.rsel))
                  s.select(Math.min(s.rsel, s.regions.length - 2))
                }}>
          <IconTrash />
        </button>
      </span>
    </div>
  )
}

/** Three decimals with the trailing zeros off, so a box at a half reads `0.5` and one
 *  mid-drag still reads to the precision the arrows step by. */
const fmt = (v: number) => v.toFixed(3).replace(/0+$/, '').replace(/\.$/, '')
