import { Glyph } from './Glyph'
import type { Kind } from '../store'

/**
 * The way into the vocabulary.
 *
 * **It carries a word, and that is the whole correction.** As a bare 34px glyph it
 * sat between the size button and `+ LoRA` announcing nothing — one of two opaque
 * marks that between them made the app's two biggest features look like a pair of
 * settings toggles. An icon can carry a control whose home you are already in; it
 * cannot announce a *destination*, and eighty-seven tiles behind one press is a
 * destination.
 *
 * **It is back in the strip rather than on the rail, and the number is why.** The
 * rail is the right *room* — these are words that land in the prompt — but giving
 * the door its own row cost 34px of a console capped at 30% of the viewport, at
 * rest, forever, to hold one button. Measured: 120px resting became 154px, while
 * the rail with pills in it only ever added 29px on top of that, because then the
 * row is carrying something. A line per button is not a price this console can pay,
 * and `+ LoRA` is already precedent for the strip hosting a door that writes into
 * the prompt.
 *
 * The tile stays, because it is the one place in this app where an icon can *teach*:
 * a dolly-out is a thing neither the word nor a static picture shows you. It animates
 * on hover and is frozen otherwise — the page at rest runs nothing, which is the same
 * rule that freezes the pills' own copies. The glyph follows the kind so the tease is
 * never a move the thing in front of you cannot make: the camera group is video-only,
 * and on the image side a framing is what a shot size means there.
 */
export function ShotDoor({ id, kind, on, onClick }: {
  id: string
  kind: Kind
  on: boolean
  onClick: (e: React.MouseEvent<HTMLButtonElement>) => void
}) {
  return (
    <button className={`opt ib shot-door${on ? ' on' : ''}`} id={id} type="button"
            title="Framing, angle, light and tone — the words this model was trained on."
            onClick={onClick}>
      <Glyph cls={kind === 'video' ? 'ca-pull' : 'fr-mcu'} />
      <span className="lead">Shot</span>
    </button>
  )
}
