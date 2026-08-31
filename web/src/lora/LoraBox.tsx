import { NumInput } from '../ui/NumInput'
import { useStore } from '../store'

/**
 * The LoRAs on this canvas, folded away until you want them.
 *
 * **Four long filenames is the case this is shaped for.** They were tokens in the
 * prompt, so four of them was four sentences' worth of markup in the middle of
 * the words you were writing; a rail of chips would be no better at four. What
 * you need most of the time is one line saying *there are four*, and the chips
 * only when you are changing one.
 *
 * So it is a disclosure and not a popover, built as one more `#shot-peek` rather
 * than as a new kind of thing. Four properties that pattern has and a popover
 * does not:
 *
 *   - **Zero pixels at rest.** Nothing to render with no LoRAs, which is the rule
 *     `#shot-rail:empty` encodes — a row is affordable when it carries content
 *     and never when it carries one control.
 *   - **In flow**, so nothing sits on top of a render. A popover floats over
 *     the canvas; this pushes the console, whose own overflow cap absorbs it —
 *     the field no longer yields to make room, since its height stopped being
 *     budget-derived. See `fieldMax.ts`.
 *   - **It does not close on scroll**, which is this codebase's standing
 *     objection to `Popover` and disqualifying for a box you type numbers into.
 *   - It is a shape already on screen.
 *
 * The count is a **word, not a pip**. The regions button failed precisely here —
 * a count riding half-outside it read as an error marker rather than as "2
 * regions".
 */
export function LoraBox() {
  const s = useStore()
  const loras = s.loras
  if (!loras.length) return null
  const open = s.loraOpen
  return (
    <div id="lora-box" className={open ? 'open' : ''}>
      <button type="button" onClick={() => s.setLoraOpen(!open)}>
        {loras.length} {loras.length === 1 ? 'LoRA' : 'LoRAs'}
      </button>
      {open && (
        <div className="chips">
          {loras.map((c) => (
            <span className="chip" key={c.path} data-lora={c.rel}>
              <span className="nm" title={c.path}>{c.rel}</span>
              {/* `fine` is 0.05 because the useful range is 1.0 to 1.4 — the same
                  reason the numeric fields carry their own step: a shift of 1.15
                  stepped by 1 leaves behind every value the model accepts. */}
              <NumInput className="val" value={String(c.strength)} fine={0.05} bigStep={0.25}
                        title="How hard this LoRA is applied."
                        onValue={(v) => {
                          const n = parseFloat(v)
                          if (Number.isFinite(n)) s.patchLora(c.path, { strength: n })
                        }} />
              <button type="button" className="x" title={`Remove ${c.rel}`}
                      onClick={() => s.dropLora(c.path)}>×</button>
            </span>
          ))}
        </div>
      )}
    </div>
  )
}
