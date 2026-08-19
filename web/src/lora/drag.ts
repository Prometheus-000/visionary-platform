/**
 * Dragging a LoRA out of the picker and onto the thing it should apply to.
 *
 * `+ LoRA` writes at the caret, which answers "which LoRA" and leaves "where" to
 * wherever the caret happened to be. A drag answers both in one gesture, and the
 * target decides the scope exactly as it already does for a photograph: onto a box
 * it is that character, onto the prompt it is the sentence, onto bare canvas it is
 * a new box holding it.
 *
 * **A private MIME type, not `text/plain`.** Three reasons, and the first is the
 * one that would have bitten:
 *
 * - Every file drop on this page is gated on `types.includes('Files')` — the scene
 *   plate, the region likeness, the video first frame, the dataset sheet. A drag
 *   carrying its own type is invisible to all of them, so none of those handlers
 *   need to learn about this one.
 * - `text/plain` would make the prompt field's *native* text drop fire as well as
 *   ours, inserting the token twice.
 * - Dragging a LoRA over another application should do nothing rather than paste a
 *   path into it.
 *
 * The payload is the volume path, which is the only field of a `LoraFile` that is
 * a key. Everything else about it is re-derived from `loraIndex` at the drop, so a
 * drag that outlives a state refresh resolves against the current volume rather
 * than against a snapshot taken when the menu opened.
 */
import type { LoraFile } from './tokens'

export const LORA_MIME = 'application/x-visionary-lora'

/**
 * The drag in flight, kept here as well as on the `DataTransfer`.
 *
 * Not a duplicate of the payload — it is the *only* way a target can name what it
 * is holding before the drop. The drag data store is in protected mode for the
 * whole of `dragover`, so `getData` there returns `''` and a box under the cursor
 * could otherwise say no more than "a LoRA". Saying `k3nan here` is the difference
 * between an affordance and a shrug, and this app's own rule for the lit state is
 * that only the target under the cursor speaks, so what it says has to be worth
 * the one caption it gets.
 *
 * Safe because a drag is one at a time and same-document by construction: this
 * type never leaves the page, so a foreign drag can never find a stale value here
 * — it fails `isLoraDrag` first.
 */
let inFlight: LoraFile | null = null

/** Set on the drag. `effectAllowed` is `copy` because the source list is not
 *  consumed — the same LoRA can be dropped into four boxes. */
export function startLoraDrag(e: React.DragEvent, l: LoraFile): void {
  e.dataTransfer.setData(LORA_MIME, l.path)
  e.dataTransfer.effectAllowed = 'copy'
  inFlight = l
}

/** What is being dragged, for a target that wants to name it while the cursor is
 *  still over it. Null between drags. */
export const draggingLora = (): LoraFile | null => inFlight

/** Cleared on `dragend`, which fires for a drag that started in this page whether
 *  it was dropped or abandoned. `App` calls this from the same listener that
 *  clears `fileOver`. */
export const endLoraDrag = (): void => { inFlight = null }

/**
 * Whether this drag is carrying a LoRA.
 *
 * Read off `types` rather than `getData`, because the drag data store is in
 * *protected mode* for the whole of `dragover` — `getData` returns `''` there and
 * only becomes readable on `drop`. So a target that wants to light up while the
 * cursor is over it has no way to ask what it is holding except by type, which is
 * the reason the type carries the meaning and the payload carries only the path.
 */
export const isLoraDrag = (e: React.DragEvent | DragEvent): boolean =>
  [...(e.dataTransfer?.types ?? [])].includes(LORA_MIME)

/** The dropped LoRA, resolved against the index as it is *now*. Null when the drag
 *  was something else, or when the file has left the volume since it was picked
 *  up — a deleted LoRA is a drop that does nothing rather than a token naming
 *  nothing, which is the failure `loraNote` exists to report. */
export function droppedLora(e: React.DragEvent, index: LoraFile[]): LoraFile | null {
  if (!isLoraDrag(e)) return null
  const path = e.dataTransfer.getData(LORA_MIME)
  return index.find((l) => l.path === path) ?? null
}
