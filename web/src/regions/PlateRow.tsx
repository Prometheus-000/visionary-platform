import { IconOutfit, IconScene } from '../icons'
import { NumInput } from '../ui/NumInput'
import { DropTile } from '../media/DropTile'
import { dataUrl, shrinkB64 } from '../media/files'
import { NEED_EDIT_LORA } from '../lora/note'
import { useStore } from '../store'
import { attached } from '../store'

export const PLATE_TITLE = {
  scene: 'Scene photo. The picture is generated inside it — lighting, perspective '
    + 'and shadows integrate.',
  outfit: 'Outfit or object photo. Transferred onto the subjects rather than pasted '
    + 'into the frame.',
} as const

/** V12's free-role sockets, in the order the graph wires them. */
const OBJECT_ROLES = ['object1', 'object2'] as const
/** One style socket. The K/V engine's tested no-leakage route is
 *  single-reference — its two-ref node is an experiment — so the row offers
 *  exactly what the backend accepts. */
const STYLE_ROLE = 'style1' as const

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
 *
 * **The object plates are the cast row's grammar on the image side.** V12's
 * sockets 3 and 4 take a photograph plus the user's own sentence about what it
 * is, and the video side already solved exactly that shape — a thumbnail, an
 * inline note, a remove — in `scene/Material`. Same classes, same gesture, so
 * the two sides read as one application. The note is not decoration: an
 * object the prompt never refers to does close to nothing, which is why the
 * backend refuses a note-less plate and the placeholder writes the sentence's
 * shape out.
 */
export function PlateRow() {
  const s = useStore()
  const objects = OBJECT_ROLES
    .map((role) => ({ role, image: attached(s.frame, role) }))
    .filter((o) => o.image)
  const free = OBJECT_ROLES.find((role) => !attached(s.frame, role))

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

      {objects.map(({ role, image }) => {
        const note = s.frame.attachments.find((a) => a.role === role)?.note ?? ''
        return (
          <div key={role} className="tref plate-obj">
            <img src={dataUrl(image!)} alt="" draggable={false} />
            <input className="trefnote" value={note} spellCheck={false}
                   id={`g-${role}-note`}
                   placeholder="what it is — a motorcycle she leans against"
                   onChange={(e) => s.notePlate(role, e.target.value)} />
            <button type="button" className="x" title="Remove this object"
                    onClick={() => s.attach('frame', role, null)}>×</button>
          </div>
        )
      })}
      {/* One inviting tile however many are held, gone only when both sockets
          are full — the well pattern, sized to a two-socket node rather than a
          growing tray. */}
      {free && (
        <DropTile id="g-drop-object" label="Object"
                  value={null} locked={!s.state?.edit_lora}
                  glyph={<span className="tplus">＋</span>}
                  title={s.state?.edit_lora
                    ? 'A photograph of a thing — a prop, a garment, a vehicle — '
                      + 'composed into the picture. Say what it is in the note.'
                    : NEED_EDIT_LORA}
                  onFile={async (f) => {
                    const b64 = await shrinkB64(f)
                    if (b64) s.attach('frame', free, b64)
                  }}
                  onClear={() => {}} />
      )}

      {/* Style by reference — a different engine from the plates: whole-frame,
          no boxes, training-free K/V injection. Never locked, because it needs
          no weight: the one tile on this row that works on a bare install. */}
      <DropTile id={`g-drop-${STYLE_ROLE}`} label="Style"
                value={attached(s.frame, STYLE_ROLE)}
                glyph={<span className="tplus">◐</span>}
                title={'A photo whose look the render should carry — style, not '
                  + 'identity or content. Whole frame; cannot combine with '
                  + 'region boxes.'}
                onFile={async (f) => {
                  const b64 = await shrinkB64(f)
                  if (b64) s.attach('frame', STYLE_ROLE, b64)
                }}
                onClear={() => s.attach('frame', STYLE_ROLE, null)} />
      {/* Shown only while a reference is attached — the region LoRA strength's
          own rule: a strength with nothing to apply to is a number that lies.
          The lever exists because 1.0 was measured pulling more than the
          grade: composition and the subject drifted toward the reference. */}
      {attached(s.frame, STYLE_ROLE) && (
        <div className="opt n" id="g-style-strength-wrap" data-lb="Strength">
          <span className="lead">Strength</span>
          <NumInput id="g-style-strength" value={s.styleStrength}
                    fine={0.05} bigStep={0.25}
                    title="How hard the style is applied. 1 is the trained strength; lower it to keep more of your prompt's own subject and framing."
                    onValue={s.setStyleStrength} />
        </div>
      )}
    </div>
  )
}
