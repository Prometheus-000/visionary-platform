import { readSize } from '../console/size'
import { useStore } from '../store'

/**
 * The frame in miniature.
 *
 * The old per-region rows needed a 32px picture of the coordinates beside each one to
 * be legible at all — that picture was the part that worked, and one row per box was
 * the part that did not. So there is one map, the same size at eight boxes as at one,
 * and it is how you reach a box when the boxes are off the render.
 *
 * Drawn at the render's own proportions, so a portrait canvas does not read as a
 * landscape one: the whole point is recognising a box by where it sits, and the frame
 * it sits in is half of that.
 *
 * The arrows carry the selection once clicking gets unreliable. At two or three boxes
 * the map is a fine target; past that they overlap, the rectangles get smaller than a
 * pointer, and stepping is the only thing that still works at any count.
 */
export function RegionMap({ onReveal }: { onReveal: () => void }) {
  const s = useStore()
  const [rw, rh] = readSize(s.img)
  const H = 30
  const W = Math.max(20, Math.min(58, Math.round(H * (rw / rh || 1.33))))
  const many = s.regions.length > 1

  const step = (d: number) => {
    if (!s.regions.length) return
    onReveal()
    s.select((s.rsel + d + s.regions.length) % s.regions.length)
  }

  return (
    <span className="rmap-wrap">
      {many && (
        <button className="rmap-step" id="r-prev" title="Previous box" aria-label="Previous box"
                type="button" onClick={(e) => { e.stopPropagation(); step(-1) }}>‹</button>
      )}
      <button id="r-map" className="rmap" type="button" style={{ width: W + 12 }}
              title="Which box you are editing. Click one to select it."
              onKeyDown={(e) => {
                if (e.key !== 'ArrowLeft' && e.key !== 'ArrowRight') return
                if (!s.regions.length) return
                e.preventDefault()
                step(e.key === 'ArrowRight' ? 1 : -1)
              }}>
        <svg viewBox={`0 0 ${W} ${H}`} aria-hidden="true">
          <rect className="fr" x=".5" y=".5" width={W - 1} height={H - 1} rx="2" />
          {s.regions.map((r, i) => {
            const x = Math.max(0, Math.min(1, r.x))
            const y = Math.max(0, Math.min(1, r.y))
            return (
              <rect key={r.id} className={`bx${i === s.rsel ? ' on' : ''}`}
                    x={(x * W).toFixed(2)} y={(y * H).toFixed(2)}
                    width={(Math.max(0, Math.min(1 - x, r.w)) * W).toFixed(2)}
                    height={(Math.max(0, Math.min(1 - y, r.h)) * H).toFixed(2)} rx="1.2"
                    onClick={(e) => {
                      e.stopPropagation()
                      // Selecting a box means adjusting it, and adjusting means
                      // seeing it — so this is also the way back onto the canvas
                      // after a render put them away.
                      onReveal()
                      s.select(i)
                    }} />
            )
          })}
        </svg>
        {/* The count, because past three boxes the map stops being able to say which
            of them you are on and a number always can. */}
        {many && <b className="rmap-at">{s.rsel + 1}/{s.regions.length}</b>}
      </button>
      {many && (
        <button className="rmap-step" id="r-next" title="Next box" aria-label="Next box"
                type="button" onClick={(e) => { e.stopPropagation(); step(1) }}>›</button>
      )}
    </span>
  )
}
