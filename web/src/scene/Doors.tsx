import { Menu } from '../ui/Menu'
import { usePopover } from '../ui/Popover'
import { Glyph } from '../shot/Glyph'
import { chipFrom, loraIndex } from '../lora/tokens'
import { useStore } from '../store'
import { SLOTS, type CastKind } from './model'

/**
 * Cast, shot and LoRA — the three doors into the composer, at the head of the rail.
 *
 * **They carry words, and the version that did not is the fault this file exists
 * to record.** They were three 22px marks on the field's trailing edge, chosen
 * because a control riding a line the field already spends costs the console no
 * height at all. Zero height bought invisibility: the owner's reading was *"this
 * feature right now is tiny icons in the far right"*, which is word for word the
 * failure CLAUDE.md already names about `#g-shot` and `#g-regional` — two glyphs
 * side by side flattening the two biggest features in the app into one confusing
 * pair, doing the one job an icon cannot do, **announcing a capability to someone
 * who does not know it exists**.
 *
 * The argument that got past that rule was that on the edge of the box you are
 * typing in, the home *is* where you are. It is wrong twice: three unlabelled
 * marks in a corner are a destination each, not a control whose value is its own
 * label — and the corner belongs to `.neg-t`, which had it first and does not
 * announce anything.
 *
 * So they take a row, and the row is affordable because it is not holding one
 * control: it is the head of the rail the cast, the pills and the LoRAs land on,
 * which is the rule stated the right way round — **a row is affordable when it
 * carries content, and never when it carries one control.**
 *
 * The mention menu is still the primary way to make a cast member. `+ Character`
 * does exactly what typing `@` does, for somebody who has not learned that yet.
 */
export function Doors({ onPalette }: {
  /** The palette is one popover shared by both strips — a second one per side
   *  would be a second place for the dimming rules to drift — so the press is
   *  handed across rather than another being mounted here. */
  onPalette: (e: React.MouseEvent<HTMLElement>) => void
}) {
  const s = useStore()
  const cast = usePopover()
  const lora = usePopover()
  const index = loraIndex(s.state)
  const on = new Set(s.loras.map((l) => l.path))

  const add = (kind: CastKind) => {
    const member = s.addCast(kind, '')
    s.setRailOpen(member.id)
  }

  return (
    <span className="tdoors">
      <button type="button" id="v-cast" className={`tdoor${cast.open ? ' on' : ''}`}
              title="Somebody or something in the scene. Typing @ in a shot does the same."
              onClick={cast.toggle}>+ Character</button>
      {/* The tile is the one place in this app where an icon can *teach* — a
          dolly-out is a thing neither the word nor a static picture shows you —
          so it keeps its glyph beside the word rather than instead of it, and it
          animates on hover and is frozen otherwise. */}
      <button type="button" id="v-shot" className={`tdoor${s.shot.length ? ' on' : ''}`}
              title="Framing, angle, light and tone — the words this model was trained on."
              onClick={onPalette}>
        <Glyph cls="ca-pull" />Shot
      </button>
      <button type="button" id="v-lora" className={`tdoor${s.loras.length ? ' on' : ''}`}
              disabled={!index.length} title="A LoRA, plugged into this clip."
              onClick={lora.toggle}>+ LoRA</button>

      {cast.open && (
        <Menu anchor={cast.anchor} onClose={cast.close}
              items={(Object.keys(SLOTS) as CastKind[]).map((k) => ({
                label: `New ${k}`,
                run: () => { add(k) },
              }))} />
      )}
      {lora.open && (
        <Menu anchor={lora.anchor} onClose={lora.close}
              items={index.map((l) => ({
                label: l.token,
                // The trigger phrase, shown and never written — a style LoRA is
                // near-invisible without its phrase, and this is the only place it
                // appears for one you did not train yourself.
                hint: l.trigger || undefined,
                on: on.has(l.path),
                run: () => { s.toggleLora(chipFrom(l)) },
              }))} />
      )}
    </span>
  )
}
