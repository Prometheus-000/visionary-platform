import { Menu } from '../ui/Menu'
import { usePopover } from '../ui/Popover'
import { useStore } from '../store'
import { applyWrite, caretAt, caretScope, caretValue } from './caret'
import { startLoraDrag } from './drag'
import { insertLora, loraIndex, parseLoras } from './tokens'

/**
 * `+ LoRA` — a picker, not a row.
 *
 * It writes `<lora:name:1>` at the caret. The button exists for discovery, because
 * you cannot type a syntax you have never seen; after that the prompt is the
 * stack, so a fifth LoRA costs the canvas nothing.
 *
 * Labelled with the token, which is the string the item is about to write. It used
 * to be `folder · filename`, which spelled the same word twice for every LoRA
 * training produced — "my_style · my_style.safetensors" — and spelled an extension
 * that is true of every row. The token is already the shortest name that points at
 * one file, so it drops both without losing the one case that needs the folder:
 * the matched Wan speed pairs, whose files are both called `high`.
 *
 * **Every row is also a drag source**, which is the half that answers *where*. A
 * click writes at the caret and so inherits wherever the caret was; a drag names
 * its own target — a box, the sentence, or the bare frame. The click is not
 * redundant: it is the only one of the two that works from a keyboard, and it is
 * the faster gesture when the caret is already where you want the token.
 */
export function LoraButton({ id }: { id: string }) {
  const state = useStore((s) => s.state)
  const prompt = useStore((s) => s.prompt)
  const pop = usePopover()
  const index = loraIndex(state)

  // Ticked means "already in the prompt", and the tick is what makes the second
  // click legible as a removal rather than a click that did nothing. Read off the
  // field the caret is in, because that is the field the pick will edit.
  const inField = new Set(
    parseLoras(index, caretScope() ? caretValue() : prompt)
      .filter((t) => t.hit).map((t) => t.hit!.path),
  )

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
                // The trigger phrase, on the row that is about to write it. A
                // style LoRA is near-invisible without its phrase, so the hint
                // is what makes the pick's write legible before it happens —
                // and the only place the phrase appears for a LoRA you did not
                // train yourself.
                hint: l.trigger || undefined,
                on: inField.has(l.path),
                drag: (e: React.DragEvent) => startLoraDrag(e, l),
                run: () => {
                  const region = caretScope() === 'region'
                  applyWrite(insertLora(
                    index, caretValue(), caretAt(), l,
                    region ? '1.3' : String(l.strength ?? 1),
                    // Never into a box — a trigger there never reaches the
                    // encoder. See `insertLora`.
                    !region,
                  ))
                },
              }))} />
      )}
    </>
  )
}
