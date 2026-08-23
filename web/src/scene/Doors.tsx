import { Menu } from '../ui/Menu'
import { usePopover } from '../ui/Popover'
import { Glyph } from '../shot/Glyph'
import { chipFrom, loraIndex } from '../lora/tokens'
import { useStore } from '../store'
import { SLOTS, type CastKind } from './model'

/**
 * Cast, shot and LoRA, on the field's trailing edge.
 *
 * **They are composition, not settings.** Sitting in the bar beside the size and
 * the checkpoint they read as settings, which is what a 34px `+ LoRA` next to a
 * 34px `Shot` had been doing — and giving the rail a row of its own to hold them
 * cost 34px of a console capped at 30% of the viewport, at rest, forever. This is
 * the third answer and the only one that costs nothing: they ride a line the field
 * is already spending, where `.neg-t` already lives.
 *
 * **The icon rule survives because the room changed.** It says a glyph can carry a
 * control whose home you are already in and cannot announce a *destination* — and
 * that was the argument for putting the word "Shot" on the door while it sat in a
 * row you scan when you already know what you want. On the edge of the box you are
 * typing in, the home *is* where you are; each of these acts on the sentence under
 * it, and the first one is the gesture you have already made by typing `@`.
 *
 * Which is the real reason this is affordable: **the mention menu is the primary
 * way in, and these are the same three doors reachable without knowing that.**
 * The cast door does exactly what typing `@` does.
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
    <span className="fedge">
      <button type="button" id="v-cast" className={`fdoor${cast.open ? ' on' : ''}`}
              title="Somebody or something in the scene. Typing @ in a shot does the same."
              onClick={cast.toggle}>@</button>
      <button type="button" id="v-shot" className={`fdoor${s.shot.length ? ' on' : ''}`}
              title="Framing, angle, light and tone — the words this model was trained on."
              onClick={onPalette}>
        <Glyph cls="ca-pull" />
      </button>
      <button type="button" id="v-lora" className={`fdoor${s.loras.length ? ' on' : ''}`}
              disabled={!index.length} title="A LoRA, plugged into this clip."
              onClick={lora.toggle}>◨</button>

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
