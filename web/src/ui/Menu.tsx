import { Popover } from './Popover'

/**
 * A list of things you can do to one object.
 *
 * One floating menu, refilled — a menu rendered inside every card would put a
 * hundred hidden subtrees in a grid that is already a hundred images, and only
 * one of them can ever be open.
 *
 * `on` is a tick rather than a highlight, and the tick is what makes a second
 * click legible as a removal rather than a click that did nothing. It is used by
 * the LoRA picker (already in the prompt) and the reference role menu (this
 * picture's role), which are the two menus where an item has a state.
 *
 * `drag` is gone with `lora/drag.ts`. It made the LoRA picker's rows drag
 * sources, so one gesture could name both which LoRA and which box — and every
 * target owns a control now, so there is no *which box* left to answer.
 */
export type MenuItem =
  | { sep: true }
  | {
      label: string
      run: () => void
      on?: boolean
      danger?: boolean
      sep?: false
      /** A dim clause after the label — what picking this will actually write
       *  beyond the label itself. The LoRA picker puts the trigger phrase here,
       *  because a row that is about to write words into your sentence should
       *  show them first. */
      hint?: string
    }

export function Menu({
  anchor,
  items,
  onClose,
}: {
  anchor: HTMLElement | null
  items: MenuItem[]
  onClose: () => void
}) {
  const checks = items.some((it) => !it.sep && it.on)
  return (
    <Popover anchor={anchor} className={`menu${checks ? ' checks' : ''}`} onClose={onClose}>
      {items.map((it, i) =>
        it.sep ? <hr key={`sep${i}`} /> : (
          <button key={it.label} type="button"
                  className={[it.danger ? 'danger' : '', it.on ? 'on' : ''].filter(Boolean).join(' ')}
                  onClick={() => {
                    // Closed first. Several of these open a dialog or a sheet of
                    // their own, and a menu still on screen behind a confirm is
                    // a menu the confirm's own outside-click will dismiss.
                    onClose()
                    it.run()
                  }}>
            {it.label}
            {it.hint && <span className="hint">{it.hint}</span>}
          </button>
        ),
      )}
    </Popover>
  )
}
