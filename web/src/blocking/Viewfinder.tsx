import { Canvas, useFrame, useThree, type ThreeEvent } from '@react-three/fiber'
import { Grid, PerspectiveCamera } from '@react-three/drei'
import { useCallback, useEffect, useRef, useState } from 'react'
import * as THREE from 'three'

import {
  compile, dims, fov, norm, SENSOR_H,
  type Cam, type Mark, type Path,
} from './derive'

/**
 * The viewfinder. One canvas, no camera object, no sliders.
 *
 * **You are the camera.** There is nothing on screen representing it because
 * you are inside it — which is why the plan view is gone rather than hidden: an
 * operator does not get an overhead, they turn around.
 *
 * **Two acts, separated by what you touch** rather than by a mode (sentence 3):
 * a body under your finger is blocking, empty space is camerawork. Nothing is
 * selected first and there is nothing to put down.
 *
 *   drag a body        move a person on the floor, at their own depth
 *   drag empty space   dolly and truck — you walk
 *   two-finger ⇔ ⇕     turn your head, both axes
 *   ⌘ two-finger ⇕     crouch and rise on the sticks
 *   pinch              change the lens
 *   double-tap a body  see through their eyes; escape to step back out
 *   ⌥ tap a body       stand them up, lay them down
 *   space              drop the camera's mark, and again to lift it
 *
 * Every one of those is something a body does. None of them needs a label,
 * which is the whole test (sentence 1) — and none is the arbitrary mapping a
 * slider forces, where dragging left somehow raises you.
 *
 * **Tilt was derived and is now a gesture, which is a correction rather than
 * an addition.** The camera used to hold whoever it was nearest so that tilt
 * fell out of where you stood — defensible, and disproved by the first real
 * reference put through it: Enter the Void's death shot opens on a man's point
 * of view looking at a ceiling, *away* from the only body on the floor. "It can
 * look away" is in the phase 6 list and auto-hold had removed the ability. So
 * turning your head is two fingers on both axes, and the odd one out is your
 * own height — which a real operator changes rarely and deliberately, so it
 * takes the modifier, and ⌘ already means geometry everywhere else here.
 */

const GREY = '#9a9a9a'
/** Stage x into three's x. The model means what a crew means — +x is stage
 *  right, screen right — and three is right-handed with +Y up, so facing +Z
 *  puts +X on your left. Flipped in the one place the world is drawn. */
const wx = (x: number) => -x
const vfov = (lens: number) => (2 * fov(lens, SENSOR_H) * 180) / Math.PI
const D = Math.PI / 180

/** Massing at the mark's own size, because a mark is no longer a standing
 *  adult by construction — a body on a floor is a low slab and a ceiling
 *  fixture is a small block in the air. **A head only where one belongs**: it
 *  is what makes facing legible through a lens, and a nose on a light fitting
 *  is the same nonsense as "the bulb, facing the lens" in the prose. Taller
 *  than it is wide is the test, which is what a standing person is. */
function Figure({ mark, lit, dim, onDown, onDouble }: {
  mark: Mark; lit: boolean; dim: boolean
  onDown: (e: ThreeEvent<PointerEvent>) => void
  onDouble: (e: ThreeEvent<MouseEvent>) => void
}) {
  const { h, w, base } = dims(mark)
  const upright = h > w
  const skin = lit ? '#e8e8e8' : dim ? '#4a4f55' : GREY
  return (
    <group position={[wx(mark.x), 0, mark.z]} rotation={[0, -(mark.yaw * D), 0]}
           onPointerDown={onDown} onDoubleClick={onDouble}>
      <mesh position={[0, base + h * (upright ? 0.36 : 0.5), 0]}>
        <boxGeometry args={[w, h * (upright ? 0.72 : 1), upright ? w * 0.55 : w * 0.3]} />
        <meshStandardMaterial color={skin} roughness={1} />
      </mesh>
      {upright && <>
        <mesh position={[0, base + h * 0.85, 0]}>
          <sphereGeometry args={[h * 0.076, 16, 12]} />
          <meshStandardMaterial color={skin} roughness={1} />
        </mesh>
        {/* the nose — the only way facing reads through a lens */}
        <mesh position={[0, base + h * 0.85, h * 0.088]}>
          <sphereGeometry args={[h * 0.026, 10, 8]} />
          <meshStandardMaterial color="#111" roughness={1} />
        </mesh>
      </>}
    </group>
  )
}

/** Drives the r3f camera from the model. Nothing is drawn for it. */
function Eye({ cam }: { cam: Cam }) {
  const ref = useRef<THREE.PerspectiveCamera>(null)
  useFrame(() => {
    const c = ref.current
    if (!c) return
    c.position.set(wx(cam.x), cam.y, cam.z)
    ;(window as unknown as Record<string, unknown>).__eye = c
    c.rotation.set((cam.pitch ?? 0) * D, -(cam.yaw * D) + Math.PI, 0, 'YXZ')
    if (c.fov !== vfov(cam.lens)) { c.fov = vfov(cam.lens); c.updateProjectionMatrix() }
    ;(window as unknown as Record<string, unknown>).__eye = c
  })
  return <PerspectiveCamera ref={ref} makeDefault near={0.05} far={200} />
}

/** Every drag resolves against the ground, because every drag is somebody
 *  moving over it — a body, or you. */
function Ground({ onHit, onEmpty }: {
  onHit: (x: number, z: number) => void
  onEmpty: (e: ThreeEvent<PointerEvent>) => void
}) {
  const { camera, raycaster, pointer } = useThree()
  const plane = useRef(new THREE.Plane(new THREE.Vector3(0, 1, 0), 0))
  const at = useRef(new THREE.Vector3())
  return (
    <mesh rotation={[-Math.PI / 2, 0, 0]} onPointerDown={onEmpty}
          onPointerMove={() => {
            raycaster.setFromCamera(pointer, camera)
            if (raycaster.ray.intersectPlane(plane.current, at.current)) {
              onHit(wx(at.current.x), at.current.z)
            }
          }}>
      <planeGeometry args={[400, 400]} />
      <meshBasicMaterial visible={false} />
    </mesh>
  )
}

type Grab = { t: 'body'; i: number } | { t: 'walk'; x: number; z: number } | null

const STAND = { h: 1.7, w: 0.5, base: 0 }
const LIE = { h: 0.4, w: 1.8, base: 0 }

const START = (): { cam: Cam; marks: Mark[] } => ({
  cam: { x: 0, z: 0, y: 1.5, yaw: 0, pitch: 0, lens: 40, on: null },
  marks: [
    { id: 1, x: -0.9, z: 3.0, yaw: 170, label: 'a woman in a red coat', ...STAND },
    { id: 2, x: 0.9, z: 4.5, yaw: 200, label: 'a man at the counter', ...STAND },
    // Not a person, so it has no front and gets no facing clause — and it is
    // 2.6m off the floor, which is the whole reason a mark carries a base.
    { id: 3, x: 0, z: 3.6, yaw: 180, label: 'a bare bulb', faces: false,
      h: 0.16, w: 0.16, base: 2.6 },
  ],
})

export function Viewfinder() {
  const [{ cam, marks }, setW] = useState(START)
  const [grab, setGrab] = useState<Grab>(null)
  /** Where the camera was when you dropped its mark. With one down, the shot
   *  is a *move* rather than a position, and the prose says both ends. */
  const [anchor, setAnchor] = useState<Cam | null>(null)
  const host = useRef<HTMLDivElement>(null)

  // Ground contact under the pointer, so walking moves the *world* under you
  // by the same amount your finger moved across it — you pull yourself along
  // rather than nudging an abstract position.
  const onHit = useCallback((x: number, z: number) => {
    if (!grab) return
    setW((w) => {
      if (grab.t === 'body') {
        return { ...w, marks: w.marks.map((m, i) => i === grab.i ? { ...m, x, z } : m) }
      }
      return { ...w, cam: { ...w.cam, x: w.cam.x - (x - grab.x), z: w.cam.z - (z - grab.z) } }
    })
  }, [grab])

  // Two fingers move *your* body; one finger moves the world. Pinch is the
  // lens, because on a camera that is what a pinch has always meant.
  useEffect(() => {
    const el = host.current
    if (!el) return
    const wheel = (e: WheelEvent) => {
      e.preventDefault()
      setW((w) => {
        if (e.ctrlKey) {  // pinch on a trackpad arrives as ctrl+wheel
          const lens = Math.min(200, Math.max(12, w.cam.lens * (1 - e.deltaY * 0.01)))
          return { ...w, cam: { ...w.cam, lens } }
        }
        const yaw = norm(w.cam.yaw - e.deltaX * 0.25)
        // ⌘ is the whole difference between moving your head and moving your
        // body, and it means geometry here exactly as it does on the canvas.
        if (e.metaKey) {
          const y = Math.min(6.0, Math.max(0.15, w.cam.y - e.deltaY * 0.006))
          return { ...w, cam: { ...w.cam, y, yaw } }
        }
        const pitch = Math.min(89, Math.max(-89,
          (w.cam.pitch ?? 0) - e.deltaY * 0.18))
        return { ...w, cam: { ...w.cam, pitch, yaw } }
      })
    }
    el.addEventListener('wheel', wheel, { passive: false })
    return () => el.removeEventListener('wheel', wheel)
  }, [])

  // "The camera has a **mark**" is sentence 8 read literally: space puts one
  // down where you stand, and everything you do after it is the far end of the
  // move rather than a new position.
  useEffect(() => {
    const key = (e: KeyboardEvent) => {
      if (e.repeat) return
      if (e.code === 'Space') {
        e.preventDefault()
        setAnchor((a) => (a ? null : cam))
      }
      // **Escape steps out of a body, because nothing else can.** Double-tap
      // was the obvious pair to double-tap-to-enter and is unreachable: once
      // you are riding somebody you are *inside* them, so there is no longer
      // anything on screen to tap. Found by driving it. Escape already means
      // "back out of this scope" on the canvas.
      if (e.code === 'Escape') {
        setW((w) => (w.cam.on ? { ...w, cam: { ...w.cam, on: null } } : w))
      }
    }
    window.addEventListener('keydown', key)
    return () => window.removeEventListener('keydown', key)
  }, [cam])

  // Measured, not assumed — see `Cam.aspect`. A ResizeObserver rather than a
  // one-off, because the window is the frame here and it changes shape.
  const [aspect, setAspect] = useState(16 / 9)
  useEffect(() => {
    const el = host.current
    if (!el) return
    const ro = new ResizeObserver(() => {
      const r = el.getBoundingClientRect()
      if (r.width > 0 && r.height > 0) setAspect(r.width / r.height)
    })
    ro.observe(el)
    return () => ro.disconnect()
  }, [])

  const live: Cam = { ...cam, aspect }
  // With a mark down the shot runs from there to here, so the *prose* is
  // compiled from the anchor and the *boxes* are drawn from where you actually
  // are — otherwise the overlay would sit on a frame you are no longer in.
  const from: Cam = anchor ? { ...anchor, aspect } : live
  const path: Path = anchor
    ? { x: cam.x, z: cam.z, y: cam.y, yaw: cam.yaw, pitch: cam.pitch,
        on: cam.on ?? null }
    : null
  const { clauses, tail, boxes } = compile(from, marks, path, 6, live)

  return (
    <div className="vf">
      <div className="stage" ref={host}
           onPointerUp={() => setGrab(null)} onPointerLeave={() => setGrab(null)}>
        <Canvas dpr={[1, 2]} style={{ background: '#0b0d10' }}>
          <Eye cam={live} />
          <ambientLight intensity={1.2} />
          <directionalLight position={[4, 9, -3]} intensity={1.1} />
          <Grid args={[200, 200]} cellSize={1} sectionSize={5} infiniteGrid
                cellColor="#1b1f24" sectionColor="#2a3038" fadeDistance={44} />
          <Ground onHit={onHit}
                  onEmpty={(e) => {
                    const p = e.point
                    setGrab({ t: 'walk', x: wx(p.x), z: p.z })
                  }} />
          {marks.map((m, i) => (
            <Figure key={m.id} mark={m} lit={grab?.t === 'body' && grab.i === i}
                    dim={cam.on === m.id}
                    onDown={(e) => {
                      e.stopPropagation()
                      // ⌥ lays them down or stands them up. Pose is not a
                      // decoration: it is the mark's dimensions, and every
                      // framing the shot reports is computed from those.
                      if (e.nativeEvent.altKey) {
                        setW((w) => ({ ...w, marks: w.marks.map((k) =>
                          k.id !== m.id ? k
                            : { ...k, ...(dims(k).h > dims(k).w ? LIE : STAND) }) }))
                        return
                      }
                      setGrab({ t: 'body', i })
                    }}
                    onDouble={(e) => {
                      e.stopPropagation()
                      // Being somebody. The camera goes to their eyeline and
                      // their body leaves the frame — you cannot see yourself,
                      // and a mark asked about from inside itself returns a
                      // distance of zero. Escape is the way back out.
                      setW((w) => {
                        const d = dims(m)
                        return { ...w, cam: { ...w.cam, on: m.id, x: m.x, z: m.z,
                                              y: d.aim, yaw: m.yaw } }
                      })
                    }} />
          ))}
          {anchor && (
            <mesh position={[wx(anchor.x), 0.03, anchor.z]}
                  rotation={[-Math.PI / 2, 0, 0]}>
              <ringGeometry args={[0.18, 0.26, 24]} />
              <meshBasicMaterial color="#fbbf24" />
            </mesh>
          )}
        </Canvas>
        <svg viewBox="0 0 100 100" preserveAspectRatio="none" className="boxes">
          {boxes.map(({ m, b }) => b && (
            <rect key={m.id} x={b.x * 100} y={b.y * 100}
                  width={b.w * 100} height={b.h * 100} />
          ))}
        </svg>
      </div>
      <pre className="prose">
        <span className="dim">[Shot 1] </span>
        {clauses.length ? <b>{clauses.join(' ')} </b>
                        : <span className="dim">nobody in frame. </span>}
        {tail.join(' ')}
      </pre>
    </div>
  )
}
