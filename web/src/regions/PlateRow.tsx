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
 * Visible whenever the composer is on images, boxes or none. A plate needs an
 * identity to compose around, and a box is only one way to hold it: with no
 * boxes drawn the backend conjures a full-canvas region out of the run's
 * first LoRA chip, so the commonest render — one character, no boxes — takes
 * a scene or an outfit directly. What still carries a gate is the note under
 * the prompt, which speaks up only when *neither* a box nor a chip exists.
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
        const a = s.frame.attachments.find((x) => x.role === role)
        const note = a?.note ?? ''
        const person = !!a?.person
        return (
          <div key={role} className="tref plate-obj">
            <img src={dataUrl(image!)} alt="" draggable={false} />
            {/* Thing or person — the node's own two free roles, and they read
                the photograph differently: an object gets held, a person gets
                "face unchanged". The pill is `.tsheet`, the row family's
                existing toggle, and it flips the note from required to
                optional because an unnamed person already has a clause where
                an unnamed object does nothing. */}
            <button type="button" className={`tsheet${person ? ' on' : ''}`}
                    id={`g-${role}-person`}
                    title={person
                      ? 'A person: their face is carried from the photo as-is. '
                        + 'Click to treat the photo as a thing instead.'
                      : 'A thing composed into the picture. Click if the photo '
                        + 'is a person — their face is then carried as-is.'}
                    onClick={() => s.personPlate(role, !person)}>
              {person ? 'person' : 'object'}
            </button>
            <input autoComplete="off" className="trefnote" value={note} spellCheck={false}
                   id={`g-${role}-note`}
                   placeholder={person
                     ? 'who they are — optional; “the drummer, on the left”'
                     : 'what it is — a motorcycle she leans against'}
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

      {/* Shown only while an edit plate is attached — the style strength's own
          rule, one control over. This is V12's edit_lora_strength, hardcoded
          at the node's 0.7 default until now: low keeps more of the photo the
          run composes into, high lets the instruction rewrite more of it, and
          past ~1 the node's own tooltip prices it at mottled texture (the
          edit and character deltas add in one forward). */}
      {(['scene', 'outfit', 'object1', 'object2'] as const)
        .some((slot) => attached(s.frame, slot)) && (
        <div className="opt n" id="g-edit-strength-wrap" data-lb="Edit">
          <span className="lead">Edit</span>
          <NumInput id="g-edit-strength" value={s.editStrength}
                    fine={0.05} bigStep={0.25}
                    title="How hard the edit layer drives the compose. 0.7 is the node's default; lower keeps more of the attached photo, higher follows the instruction harder and past ~1 the texture goes mottled."
                    onValue={s.setEditStrength} />
        </div>
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
