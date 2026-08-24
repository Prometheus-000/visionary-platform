import { IconOutfit, IconScene } from '../icons'
import { DropTile } from '../media/DropTile'
import { shrinkB64 } from '../media/files'
import { NEED_EDIT_LORA } from '../lora/note'
import { useStore } from '../store'
import { attached } from '../store'

export const PLATE_TITLE = {
  scene: 'Scene photo. The picture is generated inside it — lighting, perspective '
    + 'and shadows integrate.',
  outfit: 'Outfit or object photo. Transferred onto the subjects rather than pasted '
    + 'into the frame.',
} as const

/**
 * Every picture the image run can be given, at rest — the image side's SourceRow.
 *
 * The tiles lived only on the frame card, which is two undiscovered gestures
 * deep: a double-click nobody guesses to get a box, then a corner button to
 * reach the card. The owner filed the whole feature as "the regional feature
 * doesn't work" with the engine healthy underneath it, which is the strongest
 * form of the rule this row answers: a feature a user can't see is one that
 * doesn't exist. So the tiles moved here — moved, not copied, because two live
 * homes for one attachment is a second way to do the first thing.
 *
 * Visible whenever the composer is on images, boxes or none. The plates do
 * need a region, but a dead tile teaches nothing — the natural order is often
 * plate first ("compose into this"), boxes second, and the store has no
 * problem holding a plate while the canvas is still empty. What carries the
 * gate is the note under the prompt, which names the gesture that is missing
 * rather than refusing the drop.
 *
 * Locked, not hidden, without the edit weight — the card's original argument,
 * verbatim: an install without the edit LoRA is one download away from scene
 * and outfit transfer and had no way of learning either existed.
 */
export function PlateRow() {
  const s = useStore()
  return (
    <div className="opts" id="g-plate-sec">
      {(['scene', 'outfit'] as const).map((slot) => (
        <DropTile key={slot} id={`g-drop-${slot}`} label={slot === 'scene' ? 'Scene' : 'Outfit'}
                  value={attached(s.frame, slot)} locked={!s.state?.edit_lora}
                  glyph={slot === 'scene' ? <IconScene /> : <IconOutfit />}
                  title={s.state?.edit_lora ? PLATE_TITLE[slot] : NEED_EDIT_LORA}
                  onFile={async (f) => {
                    const b64 = await shrinkB64(f)
                    if (b64) s.attach('frame', slot, b64)
                  }}
                  onClear={() => s.attach('frame', slot, null)} />
      ))}
    </div>
  )
}
