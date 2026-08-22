import { Menu } from '../ui/Menu'
import { usePopover } from '../ui/Popover'
import { useStore } from '../store'
import { chipFrom, loraIndex } from './tokens'

/**
 * `+ LoRA` — a picker, and now only a picker.
 *
 * **It used to write.** A click put `<lora:name:1>` at the caret, followed by the
 * trigger phrase, at a strength the server suggested — three edits to your
 * sentence from one press, in a field you were in the middle of writing. Every
 * one of those has a better home: the strength is the chip's own circle, the
 * phrase is yours to place at your discretion, and the token was never anything
 * the encoder saw.
 *
 * So this adds a chip and touches no text at all. The caret machinery it needed
 * to answer *where* is gone with it, and so is the drag: a drag existed because a
 * click inherited wherever the caret happened to be, and once a region has its
 * own dropdown there is no *where* left to disambiguate.
 *
 * Labelled with the token, which is the shortest name that still points at one
 * file. It used to be `folder · filename`, which spelled the same word twice for
 * every LoRA training produced — "my_style · my_style.safetensors" — and spelled
 * an extension true of every row.
 */
export function LoraButton({ id }: { id: string }) {
  const state = useStore((s) => s.state)
  const loras = useStore((s) => s.loras)
  const toggleLora = useStore((s) => s.toggleLora)
  const pop = usePopover()
  const index = loraIndex(state)

  // Ticked means "already on the canvas", and the tick is what makes the second
  // click legible as a removal rather than a click that did nothing.
  const on = new Set(loras.map((l) => l.path))

  return (
    <>
      <button className="s" id={id} type="button" disabled={!index.length}
              style={{ height: 32, padding: '0 10px' }} onClick={pop.toggle}>
        + LoRA
      </button>
      {pop.open && (
        <Menu anchor={pop.anchor} onClose={pop.close}
              items={index.map((l) => ({
                label: l.token,
                // The trigger phrase, shown and never written. A style LoRA is
                // near-invisible without its phrase and this is the only place
                // it appears for a LoRA you did not train yourself — so it is
                // here as something to read and type where you want it, which is
                // the whole of what "nothing writes trigger words" leaves.
                hint: l.trigger || undefined,
                on: on.has(l.path),
                run: () => toggleLora(chipFrom(l)),
              }))} />
      )}
    </>
  )
}
