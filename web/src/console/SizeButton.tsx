import { Popover, usePopover } from '../ui/Popover'
import { NumInput } from '../ui/NumInput'
import { IconSwap } from '../icons'
import { useStore, videoModel } from '../store'
import {
  BUCKETS, SIZE_SCALES, commitBoxes, nearestBucket, ratio, readSize, scaleKey,
  sizeLabel, swapSize, withAspect, withScale,
} from './size'
import { resolveVid } from './resolve'

/** A rectangle drawn at its own proportions inside a fixed box. It is the one
 *  representation of an aspect ratio that needs no reading, which is the same
 *  argument the shot tiles make for a dolly-out. */
function Tile({ w, h, label, on, onPick }: {
  w: number; h: number; label: string; on: boolean; onPick: () => void
}) {
  const long = 26
  const sw = w >= h ? long : Math.round((long * w) / h)
  const sh = h > w ? long : Math.round((long * h) / w)
  return (
    <button className={`ar${on ? ' on' : ''}`} type="button" title={label} onClick={onPick}>
      <i style={{ width: sw, height: sh }} />
      <b>{label}</b>
    </button>
  )
}

/**
 * "16:9 · 2304×1296" — the value, so the control needs no name.
 *
 * A view over the store, never a second copy of it: every click writes the aspect,
 * the scale or the boxes, and the frame, the payload and the arrow keys all read
 * those. Rebuilt on each change rather than patched, because at eight tiles and
 * four scales the diff is more code than the redraw and one of them can be wrong.
 */
export function SizeButton() {
  const img = useStore((s) => s.img)
  const setImg = useStore((s) => s.setImg)
  const pop = usePopover()
  const [w, h] = readSize(img)
  const k = scaleKey(img.scale)

  return (
    <>
      <button className={`opt ib${pop.open ? ' on' : ''}`} id="g-size" type="button"
              title="Shape and resolution" onClick={pop.toggle}>
        {sizeLabel(img)}
      </button>
      {pop.open && (
        <Popover anchor={pop.anchor} className="menu sizer" onClose={pop.close}>
          <div className="ars">
            {BUCKETS.map(([v, label]) => {
              const [bw, bh] = v.split('x').map(Number)
              return (
                <Tile key={v} w={bw ?? 1} h={bh ?? 1} label={label} on={v === img.aspect}
                      onPick={() => setImg(withAspect(img, v))} />
              )
            })}
          </div>
          <div className="scales">
            {SIZE_SCALES.map(([v, label]) => (
              <button key={v} type="button"
                      className={`sc${v === k && img.aspect !== 'custom' ? ' on' : ''}`}
                      onClick={() => {
                        // Choosing a scale on a custom size is how you get back to
                        // a bucket: it has to pick one, or the button would light
                        // with nothing behind it.
                        const base = img.aspect === 'custom'
                          ? { ...img, ...withAspect(img, nearestBucket(w, h)) }
                          : img
                        setImg(withScale(base, v))
                      }}>
                {label}
              </button>
            ))}
          </div>
          {/* The raw store value, never `readSize`'s.
            *
            * `readSize` floors to 8 and clamps to 64–2048, which is right for
            * the payload, the label and the frame — and wrong for the box you
            * are typing in, because it applies on every render. Fed back as the
            * input's value it snapped per keystroke: "1" became 64, and the
            * caret then walked through a box that kept rewriting itself, ending
            * at the 2048 clamp. Typing 1153 must show 1153 until you leave the
            * field. `commitBoxes` on blur and Enter is what snaps it, which is
            * the whole distinction between a value and a committed one. */}
          <div className="sz-custom">
            <label>
              W
              <NumInput value={String(img.width)} inputMode="numeric" bigStep={8}
                        onValue={(v) => setImg({ aspect: 'custom', width: Number(v) || 0 })}
                        onCommit={() => setImg(commitBoxes(img))} />
            </label>
            <button className="sz-swap" type="button" title="Swap width and height"
                    onClick={() => setImg(swapSize(img))}>
              <IconSwap />
            </button>
            <label>
              H
              <NumInput value={String(img.height)} inputMode="numeric" bigStep={8}
                        onValue={(v) => setImg({ aspect: 'custom', height: Number(v) || 0 })}
                        onCommit={() => setImg(commitBoxes(img))} />
            </label>
          </div>
          <p className="muted sz-note">
            {img.aspect === 'custom'
              ? `Custom${ratio(w, h) ? ` ${ratio(w, h)}` : ''} — snapped to 8, the VAE grid.`
              : `${BUCKETS.find(([v]) => v === img.aspect)?.[1] ?? ''} at `
                + `${SIZE_SCALES.find(([v]) => v === k)?.[1] ?? ''} · `
                + `${((w * h) / 1e6).toFixed(1)} MP`}
          </p>
        </Popover>
      )}
    </>
  )
}

/**
 * The video side's size control.
 *
 * Deliberately its own component rather than a prop on the one above: the image
 * side addresses trained pixel buckets and a scale multiplier, the video side
 * addresses ratio strings and a tier key whose labels come from the chosen model.
 * Same shape on screen and the same `.sizer` styles, different vocabulary
 * underneath — folding them together would mean one component branching on which
 * of two unrelated things it holds.
 *
 * The video side had no size control at all for a while: a separate aspect select
 * and tier select, 1016px of strip between them.
 */
export const VIDEO_ASPECTS = ['21:9', '16:9', '4:3', '1:1', '3:4', '9:16']

export function VideoSizeButton() {
  const s = useStore()
  const pop = usePopover()
  const m = videoModel(s)
  const r = resolveVid(s)
  const tiers = Object.entries(m?.tiers ?? {})
  // A first frame anchors the geometry — the canvas follows the image, so the
  // aspect stops being the thing that decides it. Disabling it and saying why
  // beats leaving a control that looks live and is quietly ignored.
  const locked = !!s.keyframe.first

  return (
    <>
      <button className={`opt ib${pop.open ? ' on' : ''}`} id="v-size" type="button"
              title={locked
                ? 'The first frame sets the shape. Resolution only.'
                : 'Shape and resolution'}
              onClick={pop.toggle}>
        {[locked ? 'first frame' : s.vid.aspect, m?.tiers?.[r.tier]].filter(Boolean).join(' · ') || 'Size'}
      </button>
      {pop.open && (
        <Popover anchor={pop.anchor} className="menu sizer" onClose={pop.close}>
          {!locked && (
            <div className="ars">
              {VIDEO_ASPECTS.map((a) => {
                const [aw, ah] = a.split(':').map(Number)
                return (
                  <Tile key={a} w={aw ?? 1} h={ah ?? 1} label={a} on={a === s.vid.aspect}
                        onPick={() => s.setVid({ aspect: a })} />
                )
              })}
            </div>
          )}
          <div className="scales">
            {tiers.map(([key, label]) => (
              <button key={key} type="button" className={`sc${key === r.tier ? ' on' : ''}`}
                      onClick={() => s.setVid({ tier: key })}>
                {label}
              </button>
            ))}
          </div>
          {locked && (
            <p className="muted sz-note">
              The clip follows the first frame’s aspect ratio.
            </p>
          )}
        </Popover>
      )}
    </>
  )
}
