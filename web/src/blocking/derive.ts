/**
 * The blocking derivation, mirroring `_stage_*` in app.py.
 *
 * A mirror on purpose, and the same bargain `storyline/model.ts` already
 * strikes: `/api/compile` is the authority and this exists so a drag can answer
 * at 60fps rather than over the network. **The band tables below are copied
 * from app.py verbatim.** If the two ever disagree, this one is wrong.
 *
 * Metres on a ground plane: `x` lateral, `z` away from the camera, `y` height.
 * `yaw` in degrees, 0 looking along +z.
 */

export const SENSOR_W = 36, SENSOR_H = 24
export const FIG_H = 1.7, FIG_W = 0.5, EYE_H = 1.6
/** Tilt below which the camera sees the body it rides. Anatomy, not taste:
 *  eyes sit above and in front of the chest, so a level or raised gaze holds
 *  none of you. It matters because **POV is legible only when your own limbs
 *  are in frame** — a video model shown Enter the Void's POV act, which looks
 *  up at a ceiling, read it as a free camera move, correctly on the pixels. */
export const OWN_BODY = -25

/** Frame-height fractions. A person 2.4x the frame's height is a close-up. */
const SIZE: [number, string][] = [
  [4, 'in an extreme close-up'], [2.4, 'in a close-up'],
  [1.5, 'in a medium close-up'], [1, 'in a medium shot'],
  [0.45, 'in a wide shot'], [0, 'in an extreme wide shot'],
]
const ANGLE: [number, string][] = [
  [45, "shot from directly overhead, a bird's-eye view"],
  [7, 'shot from a high angle'], [-7, 'shot at eye level'],
  [-45, 'shot from a low angle'],
  [-181, "shot from ground level looking up, a worm's-eye view"],
]
/** Descending, like every table here — `band` returns the first row the value
 *  clears, so an ascending one silently returns its last. Written ascending,
 *  this reported somebody looking down the barrel as back to the lens. */
const FACING: [number, string][] = [
  [155, 'with their back to the lens'],
  [115, 'turned three-quarters away from the lens'],
  [65, 'in profile to the lens'],
  [25, 'turned three-quarters toward the lens'],
  [0, 'facing the lens'],
]
const NEAR: [number, string][] = [
  [3, 'across the space from {other}'], [1.2, 'a few steps from {other}'],
  [0.6, "within arm's reach of {other}"],
  [0, 'close enough that their shoulders overlap'],
]
const VERB: Record<string, string> = {
  pushin: 'pushes in', pullout: 'pulls out', panl: 'pans left',
  panr: 'pans right', tiltu: 'tilts up', tiltd: 'tilts down',
  truckl: 'trucks left', truckr: 'trucks right',
  craneu: 'cranes up', craned: 'cranes down',
  arc: 'arcs around the subject', static: 'holds a static shot',
}

/** `pitch` is derived, never set: the camera holds whoever it is nearest, so
 *  rising tilts down to keep them framed. Tilt is a consequence of where you
 *  stand, not a control — sentence 8, the camera as a character that can look
 *  away rather than a settings group. */
export type Cam = {
  x: number; z: number; y: number; yaw: number; lens: number; pitch?: number
  /** The mark this camera *is*. Not derivable and not verifiable from a
   *  render — a video model cannot tell a camera at a man's eye from a camera
   *  that happens to be where his eye is, because there is no visual
   *  difference. So it is authored, never inferred. */
  on?: number | null
  /** The frame's width/height, **measured from the surface it is drawn on**
   *  rather than assumed from the sensor.
   *
   *  Vertical field comes from the 24mm sensor height and the lens; horizontal
   *  is that times the aspect, which is what a camera actually does when you
   *  change frame shape — it crops sideways. Three derives its own horizontal
   *  field from the canvas, so hard-coding 36/24 here made the two disagree
   *  every time the viewport changed shape, and drew every box beside the body
   *  it described. Reading it back closes that off for good. */
  aspect?: number
}
/** A mark is not a standing adult. `FIG_*` are defaults now: a body curled on
 *  a floor is 0.4 x 1.8 at base 0, a ceiling fixture 0.15 x 0.15 at base 2.7,
 *  and both were subjects in the reference that found this. `faces` is off for
 *  anything without a front — "the bulb, facing the lens" is the band that
 *  earns this feature answering a question nobody asked of it. */
export type Mark = {
  id: number; x: number; z: number; yaw: number; label: string
  h?: number; w?: number; base?: number; faces?: boolean
}
export const dims = (m: { h?: number; w?: number; base?: number }) => {
  const h = m.h ?? FIG_H, w = m.w ?? FIG_W, base = m.base ?? 0
  return { h, w, base, aim: base + h * (EYE_H / FIG_H) }
}
export type Path = {
  x?: number; z?: number; y?: number; yaw?: number; pitch?: number
  /** Present-and-null lets the camera stop being somebody mid-shot. */
  on?: number | null
} | null

const D = Math.PI / 180
const deg = (r: number) => r / D
export const norm = (d: number) => {
  const v = (((d + 180) % 360) + 360) % 360 - 180
  return v <= -180 ? v + 360 : v
}
const band = (v: number, t: [number, string][]) =>
  t.find(([e]) => v >= e)?.[1] ?? t[t.length - 1]?.[1] ?? ''
export const fov = (lens: number, sensor: number) =>
  Math.atan(sensor / (2 * Math.max(1, lens)))

/** Horizontal half-angle. Falls back to the 36x24 sensor's own 3:2 when no
 *  surface has reported its shape, so the maths still stands alone. */
const hFov = (cam: { lens: number; aspect?: number }, hv: number) =>
  cam.aspect ? Math.atan(Math.tan(hv) * cam.aspect) : fov(cam.lens, SENSOR_W)

export type Seen = {
  dist: number; flat: number; sx: number; sy: number; fill: number
  fw: number; fh: number; pitch: number
  facing: number; h: number; w: number; base: number; aim: number
  behind: boolean; inFrame: boolean; size: string; angle: string
}

export function see(cam: Cam, m: Parameters<typeof dims>[0] &
                    { x: number; z: number; yaw: number }): Seen {
  const { h, w, base, aim } = dims(m)
  const dx = m.x - cam.x, dz = m.z - cam.z
  // Two distances, and conflating them cost a whole shot. `flat` is the plan
  // view, which bearing and pitch are measured against; `dist` is the real
  // one. Size used to read `flat`, so craning three metres straight up over
  // somebody left the framing reading exactly as it did from the floor.
  const flat = Math.hypot(dx, dz)
  const dist = Math.hypot(flat, aim - cam.y)
  const bearing = norm(deg(Math.atan2(dx, dz)) - cam.yaw)
  const hv = fov(cam.lens, SENSOR_H)
  // **The measured horizontal field, not the 36mm one.** `box` used `hFov` and
  // this used the sensor, so the prose and the rectangles were computed on two
  // different frames and the overlay sat beside the bodies it described. Only
  // visible once the two frame tests were collapsed into one — until then each
  // was self-consistent and they were quietly measuring different pictures.
  const hh = hFov(cam, hv)
  // Behind the camera reads as off-frame rather than as a wrapped angle, which
  // `tan` would do silently and put somebody back in shot facing backwards.
  const behind = Math.abs(bearing) >= 90
  const sx = behind ? 9.9 : Math.tan(bearing * D) / Math.tan(hh)
  // Whichever axis it spans most. Height alone was right while every mark was
  // a standing figure and reads a body lying down as a wide shot from a metre.
  const d = Math.max(0.05, dist)
  const fw = w / (2 * d * Math.tan(hh)), fh = h / (2 * d * Math.tan(hv))
  const fill = Math.max(fh, fw)
  // Vertical screen position off the camera's own axis, -1 the top edge. There
  // was none while every lens pointed at the horizon, and the two frame tests
  // that grew up in its absence disagreed the moment either changed.
  const riseC = Math.atan2(base + h / 2 - cam.y, Math.max(0.05, flat))
  const sy = -Math.tan(riseC - (cam.pitch ?? 0) * D) / Math.tan(hv)
  // Deliberately the *standing* angle, not the tilt: "shot from a low angle"
  // is a fact about where the camera is, and a tripod head tilting to keep
  // somebody framed does not change it.
  const pitch = deg(Math.atan2(cam.y - aim, Math.max(0.05, flat)))
  const toCam = deg(Math.atan2(-dx, -dz))
  return {
    dist, flat, h, w, base, aim, sx, sy, fill, fw, fh, pitch,
    facing: Math.abs(norm(m.yaw - toCam)),
    behind,
    // In frame if any of it is, on both axes — a centre-based test puts a
    // close-up's midpoint below the bottom edge and calls them absent. But
    // **behind is behind**: `sx` carries 9.9 as a sentinel rather than a
    // coordinate, and something close enough subtends more than 8.9 frames, so
    // allowing it its own width either side let a body the camera was standing
    // inside come back in frame.
    inFrame: !behind && Math.abs(sx) <= 1 + fw && Math.abs(sy) <= 1 + fh,
    size: band(fill, SIZE), angle: band(pitch, ANGLE),
  }
}

export type Box = { x: number; y: number; w: number; h: number; seen: Seen }

/** A mark projected onto the image plane — the rectangle a region already is. */
export function box(cam: Cam, m: Mark): Box | null {
  const s = see(cam, m)
  // **The frame test lives in `see` and nowhere else.** This carried its own
  // and the two disagreed the moment a camera could tilt.
  if (!s.inFrame) return null
  const w = s.fw, h = s.fh
  // Screen y runs *down*, which is the sign that is easy to invert and puts
  // everybody on the ceiling. It comes off `sy` now rather than being derived
  // a second time here — same reason as the frame test.
  const cy = (s.sy + 1) / 2
  const cx = (s.sx + 1) / 2
  const x = Math.min(Math.max(cx - w / 2, 0), 1)
  const y = Math.min(Math.max(cy - h / 2, 0), 1)
  const bw = Math.min(w, 1 - x), bh = Math.min(h, 1 - y)
  return bw > 0 && bh > 0 ? { x, y, w: bw, h: bh, seen: s } : null
}

const where = (s: Seen) => {
  // Which way out, now that there are two ways. Left and right were the only
  // answers while inFrame was horizontal alone, so a ceiling fixture above the
  // top edge was reported as off to one side.
  if (!s.inFrame) {
    if (Math.abs(s.sx) > 1 + s.fw) {
      return s.sx < 0 ? 'just off-frame left' : 'just off-frame right'
    }
    return s.sy < 0 ? 'just above the frame' : 'just below the frame'
  }
  const side = s.sx < -0.33 ? 'screen left' : s.sx > 0.33 ? 'screen right' : 'centre frame'
  const depth = s.fill >= 1.5 ? 'in the foreground' : s.fill < 0.45 ? 'in the background' : ''
  return depth ? `${side}, ${depth}` : side
}

export function moveKey(cam: Cam, p: NonNullable<Path>) {
  const dx = (p.x ?? cam.x) - cam.x, dz = (p.z ?? cam.z) - cam.z
  const dy = (p.y ?? cam.y) - cam.y
  const dyaw = norm((p.yaw ?? cam.yaw) - cam.yaw)
  const dtilt = norm((p.pitch ?? cam.pitch ?? 0) - (cam.pitch ?? 0))
  const yaw = cam.yaw * D, tilt = (cam.pitch ?? 0) * D
  const plan = dx * Math.sin(yaw) + dz * Math.cos(yaw)
  const lat = dx * Math.cos(yaw) - dz * Math.sin(yaw)
  // The camera's own frame, all three axes of it. Vertical used to be compared
  // against the *world's* up, which is the same thing only while the lens
  // points at the horizon: a camera aimed at the floor and dropping toward a
  // body on it travels along its own forward axis, and the world-axis test
  // called that a crane down instead of a push in.
  const fwd = plan * Math.cos(tilt) + dy * Math.sin(tilt)
  const rise = dy * Math.cos(tilt) - plan * Math.sin(tilt)
  const travel = Math.hypot(Math.hypot(dx, dz), dy)
  if (travel < 0.05 && Math.abs(dyaw) < 3 && Math.abs(dtilt) < 3) return 'static'
  // Turning on the spot, on whichever axis turned further. `tiltu`/`tiltd` were
  // unreachable while this read yaw alone.
  if (travel < 0.05) {
    if (Math.abs(dtilt) > Math.abs(dyaw)) return dtilt > 0 ? 'tiltu' : 'tiltd'
    return dyaw > 0 ? 'panr' : 'panl'
  }
  if (Math.abs(rise) > Math.max(Math.abs(fwd), Math.abs(lat))) return rise > 0 ? 'craneu' : 'craned'
  if (Math.abs(lat) > Math.abs(fwd) && Math.abs(dyaw) > 12) return 'arc'
  if (Math.abs(lat) > Math.abs(fwd)) return lat > 0 ? 'truckr' : 'truckl'
  return fwd > 0 ? 'pushin' : 'pullout'
}

/** With nobody in frame there is no amplitude, only a speed — amplitude is
 *  measured against how far away the subject is, and there isn't one. */
function moveNote(cam: Cam, p: NonNullable<Path>, dist: number | null, secs: number) {
  const dx = (p.x ?? cam.x) - cam.x, dz = (p.z ?? cam.z) - cam.z
  const dy = (p.y ?? cam.y) - cam.y
  const travel = Math.hypot(Math.hypot(dx, dz), dy)
  if (travel < 0.05) return ''
  // Amplitude is relative to how far the subject is: a metre is a large move
  // at two metres and nothing at twenty. Speed is metres per second over the
  // shot, which is the one place the duration is load-bearing.
  const rate = travel / Math.max(0.5, secs)
  const sp = rate > 0.8 ? 'fast' : rate < 0.25 ? 'slow' : 'moderate'
  if (dist === null) return `at ${sp} speed`
  const r = travel / Math.max(0.5, dist)
  const amp = r > 0.5 ? 'large' : r < 0.18 ? 'small' : 'medium'
  return `with ${amp} amplitude at ${sp} speed`
}

/**
 * The clauses, ordered left to right across the frame.
 *
 * `_shot_body` already records that subjects come out of the model in the
 * order they are described; until blocking existed that order was whatever
 * somebody happened to type. The relation **trails**, which is the rule the
 * compiler already keeps: the subject opens the clause and how they stand to
 * the others closes it.
 */
/** The marks the camera can see, which is every mark except the one it *is*.
 *  Asking `see` about the body you are riding returns a distance of zero, which
 *  the lead picker reads as the nearest thing in the room. */
export const others = (cam: Cam, marks: Mark[]) =>
  cam.on ? marks.filter((m) => m.id !== cam.on) : marks

/** Framing and angle from one camera position. POV is a framing value like any
 *  other, which is what lets a shot open in one and end out of it; the angle
 *  still comes off the lead, so a camera on a bathroom floor looking up at a
 *  fixture is a POV shot *and* a worm's-eye view. */
export function read(cam: Cam, marks: Mark[]): [string | null, string | null] {
  // Nothing in frame means no lead, and no lead means no framing. Falling back
  // to the nearest mark overall started answering with whoever was *behind*
  // the lens once inFrame was true on both axes.
  const framed = others(cam, marks).map((m) => see(cam, m)).filter((s) => s.inFrame)
  const lead = framed.length
    ? framed.reduce((a, s) => (s.fill > a.fill ? s : a)) : null
  // The phrase, not the key. `SIZE` and `ANGLE` hold phrases, so returning a
  // bare 'pov' beside them printed "Pov." where everything else reads as a
  // clause — the one value in this file that was not already a sentence.
  return [cam.on ? 'in a first-person point-of-view shot' : (lead?.size ?? null),
          lead?.angle ?? null]
}

/** The camera where the path leaves it. `on: null` present-and-null is the
 *  detach — one continuous take that is a man's point of view until his soul
 *  leaves the body is a camera letting go of a mark, not a cut. */
export function endCam(cam: Cam, path: Path): Cam {
  if (!path) return cam
  return {
    ...cam,
    x: path.x ?? cam.x, z: path.z ?? cam.z, y: path.y ?? cam.y,
    yaw: path.yaw ?? cam.yaw, pitch: path.pitch ?? cam.pitch,
    ...(path.on !== undefined ? { on: path.on } : {}),
  }
}

export function compile(cam: Cam, marks: Mark[], path: Path, secs: number,
                        viewCam?: Cam) {
  const seen = others(cam, marks).map((m) => ({ m, s: see(cam, m) }))
  const on = seen.filter((o) => o.s.inFrame).sort((a, b) => a.s.sx - b.s.sx)
  const label = (m: Mark) => `<Subject ${marks.indexOf(m) + 1}>`
  const clauses = on.map(({ m, s }) => {
    const bits = [`${label(m)} ${where(s)}`]
    if (m.faces !== false) bits.push(band(s.facing, FACING))
    const others = seen.filter((o) => o.m !== m)
    if (others.length) {
      // In three dimensions. Two people stand on the same floor, so the plan
      // distance was the whole answer while every mark was a person — a bulb
      // 1.1m away across the floor and 2.6m straight up came out as "within
      // arm's reach of the bulb".
      const gap = (o: Mark) => Math.hypot(
        Math.hypot(o.x - m.x, o.z - m.z), dims(o).aim - dims(m).aim)
      const near = others.reduce((a, o) => (gap(o.m) < gap(a.m) ? o : a))
      bits.push(band(gap(near.m), NEAR).replace('{other}', label(near.m)))
    }
    return `${bits.join(', ')}.`
  })
  // The rider's own body leads: nearest thing in the shot, and the only thing
  // that says whose eyes these are. Excluding the rider outright was right
  // about their face and wrong about the rest of them.
  if (cam.on && (cam.pitch ?? 0) < OWN_BODY) {
    const who = marks.find((m) => m.id === cam.on)
    if (who) clauses.unshift(`${label(who)}'s own arms and torso across the ` +
      `bottom of the frame, seen from their own eyes.`)
  }
  const framed = on
  // Prominence, not proximity. Nearest was the rule while every mark was a
  // person and inverts the moment they are not the same size: a bare bulb half
  // a metre off the lens is nearer than the body three metres below it, so
  // "nearest" called an overhead shot of a dying man a wide shot of a fixture.
  const lead = framed.length
    ? framed.reduce((a, o) => (o.s.fill > a.s.fill ? o : a))
    : null
  const tail: string[] = []
  // **Only when the two ends agree.** A steady state is not true of a shot
  // whose framing changes, so stating the opening one was describing the first
  // frame and calling it the shot.
  const a = read(cam, marks), b = read(endCam(cam, path), marks)
  const held = [a[0] === b[0] ? a[0] : null, a[1] === b[1] ? a[1] : null]
  if (held[0] || held[1]) {
    const t = `${held.filter(Boolean).join(', ')}.`
    tail.push(t.charAt(0).toUpperCase() + t.slice(1))
  }
  // A camera move is a fact about the camera, so it is stated whether or not
  // anybody is in frame. Gating it on a lead is what made the probe go silent
  // when the mark went down on an empty frame.
  if (path) {
    const k = moveKey(cam, path)
    const n = moveNote(cam, path, lead ? lead.s.dist : null, secs)
    tail.push(k === 'static'
      ? 'The camera holds a static shot.'
      : `The camera ${VERB[k]}${n ? ` ${n}` : ''}.`)
    const moved = [0, 1].filter((i) => a[i] && b[i] && a[i] !== b[i])
    if (moved.length) {
      const side = (r: typeof a) => moved.map((i) => r[i]).join(', ')
      tail.push(`Opens ${side(a)} and ends ${side(b)}.`)
    }
  }
  const vc = viewCam ?? cam
  // `others`, not `marks` — the body the camera is does not get a rectangle,
  // for the same reason it does not get a clause. Python's projection always
  // filtered here and this one did not, so riding a mark drew a box over the
  // whole frame for the person wearing the camera.
  return { clauses, tail, boxes: others(vc, marks).map((m) => ({ m, b: box(vc, m) })) }
}
