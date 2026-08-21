import { Glyph } from '../shot/Glyph'

/**
 * The video side's door, where the shot door used to be.
 *
 * Same skeleton as `ShotDoor` — a word and an animated glyph in the strip —
 * because it is the same kind of thing: a door that writes into the prompt,
 * beside `+ LoRA` which already does. What changed is what comes out of it.
 * The palette's tiles were comma-separated phrases in a vacuum, with no
 * knowledge of what they were acting on; behind this door the model has
 * *looked at the frame*, and every suggestion is a sentence about the subjects
 * actually in it.
 *
 * It lights when anything it manages is in play — a picked motion sentence or
 * an audio pill on the rail — for the reason the shot door lit on pills: the
 * door is the only place that can say "this run carries more than the box
 * shows at a glance".
 */
export function MotionDoor({ id, on, onClick }: {
  id: string
  on: boolean
  onClick: (e: React.MouseEvent<HTMLButtonElement>) => void
}) {
  return (
    <button className={`opt ib shot-door${on ? ' on' : ''}`} id={id} type="button"
            title="What could move in this frame — and what it sounds like."
            onClick={onClick}>
      <Glyph cls="ca-pull" />
      <span className="lead">Motion</span>
    </button>
  )
}
