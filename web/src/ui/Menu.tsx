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
 * `drag` makes an item a *source* as well as a command — the LoRA picker's rows,
 * which can be dropped on a box, the prompt or the bare frame instead of being
 * clicked. It is opt-in per item rather than a property of the menu because the
 * other two menus here act on an object that is already selected: there is
 * nowhere to carry them to.
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
      /** Written onto the drag as one private MIME type. See `lora/drag.ts`. */
      drag?: (e: React.DragEvent) => void
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
                  className={[it.danger ? 'danger' : '', it.on ? 'on' : '',
                              it.drag ? 'draggable' : ''].filter(Boolean).join(' ')}
                  draggable={!!it.drag}
                  onDragStart={it.drag && ((e) => {
                    it.drag!(e)
                    // Deferred by a tick, and that tick is the whole of it. The menu
                    // covers the canvas it is anchored over, so leaving it up means
                    // dragging a LoRA across an opaque panel to reach a box it is
                    // hiding. Closing *synchronously* inside dragstart unmounts the
                    // source node before the browser has snapshotted its drag image,
                    // which cancels the drag outright in WebKit. One turn of the loop
                    // is after the snapshot and long before the first `dragover`.
                    setTimeout(onClose, 0)
                  })}
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
