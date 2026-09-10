/**
 * One group of the served vocabulary, as tiles, for a panel.
 *
 * The palette's own tiles and the palette's own sheet (`.pal`), one section
 * deep: a storyboard panel picks a framing, an angle and a camera move the
 * way the composer does, from the same table, because the panel *is* a shot's
 * intent and the hand-off has to be a copy. For the camera the guide's two
 * other dimensions sit under the tiles — amplitude and speed — so "a large,
 * fast pan" is two more presses and the stencil on the frame grows to match.
 */
import type { CameraAmp, CameraSpeed, ShotGroup, ShotPill } from '../api/types'
import { Glyph } from '../shot/Glyph'
import { Popover } from '../ui/Popover'
import { AMPS, SPEEDS } from './model'

export function PillPicker({ anchor, group, pill, onPick, onClose }: {
  anchor: HTMLElement | null
  group: ShotGroup
  pill: ShotPill | null
  onPick: (p: ShotPill | null) => void
  onClose: () => void
}) {
  const key = pill?.key.split('.').slice(1).join('.') ?? null
  const item = group.items.find((it) => it.key === key)
  const camera = group.key === 'camera'
  return (
    <Popover anchor={anchor} className="pal sbpal" onClose={onClose}>
      <section>
        <h4>{group.label}</h4>
        <div className="tiles">
          <button type="button" className={`tl${!pill ? ' on' : ''}`}
                  title="Nothing chosen" onClick={() => { onPick(null); onClose() }}>
            <Glyph cls="" />
            none
          </button>
          {group.items.map((it) => (
            <button key={it.key} type="button"
                    className={`tl${key === it.key ? ' on' : ''}`}
                    title={it.phrase || ''}
                    onClick={() => {
                      const next: ShotPill = { key: `${group.key}.${it.key}` }
                      // Amplitude and speed belong to a move, so they carry
                      // across a change of move but never onto handheld or a
                      // locked-off camera, which have neither.
                      if (camera && it.verb) {
                        next.amp = pill?.amp ?? 'medium'
                        next.speed = pill?.speed ?? 'normal'
                      }
                      onPick(next)
                      if (!camera) onClose()
                    }}>
              <Glyph cls={it.glyph} />
              {it.label}
            </button>
          ))}
        </div>
      </section>
      {camera && pill && item?.verb && (
        <section className="sbdims">
          <h4>Amplitude <i>how far the frame changes</i></h4>
          <div className="seg" role="radiogroup">
            {AMPS.map((a) => (
              <button key={a} type="button" className={`s${(pill.amp ?? 'medium') === a ? ' on' : ''}`}
                      onClick={() => onPick({ ...pill, amp: a as CameraAmp })}>
                {a}
              </button>
            ))}
          </div>
          <h4>Speed <i>how fast it gets there</i></h4>
          <div className="seg" role="radiogroup">
            {SPEEDS.map((sp) => (
              <button key={sp} type="button" className={`s${(pill.speed ?? 'normal') === sp ? ' on' : ''}`}
                      onClick={() => onPick({ ...pill, speed: sp as CameraSpeed })}>
                {sp}
              </button>
            ))}
          </div>
        </section>
      )}
    </Popover>
  )
}
