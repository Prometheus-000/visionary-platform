import { Canvas, useFrame, useThree, type ThreeEvent } from '@react-three/fiber'
import { Grid, PerspectiveCamera } from '@react-three/drei'
import { useMemo, useRef } from 'react'
import * as THREE from 'three'

import {
  compile, fov, norm, see, SENSOR_H, FIG_H, FIG_W,
  type Cam, type Mark, type Path,
} from './derive'

/**
 * Blocking, on a real stage.
 *
 * **Two renders of one world.** The left canvas is the shot camera looking
 * through its own lens; the right is a free camera above the set with that
 * camera drawn as an object you can grab. Same marks, same positions, no
 * second data model — which is the whole argument for r3f here: the projection,
 * the occlusion and the depth sort are the renderer's job, not ours, and the
 * frame preview is literally a second camera rather than a diagram of one.
 *
 * The rejected floor plan in `master plan/what I rejected` labelled itself
 * `NO CAMERA VIEW`, which is the flaw this fixes: a blocking surface that
 * cannot show the resulting frame is blind.
 *
 * Drawn as previz — grey massing blocks, one light, no materials. It is a
 * blocking tool that happens to be 3D, not a 3D tool.
 */

const GREY = '#9a9a9a'

/**
 * Stage x, into three's x.
 *
 * The model means what a film crew means: **+x is stage right, which is screen
 * right**, looking along +z. three is right-handed with +Y up, so facing +Z
 * puts +X on your *left* — the two conventions are mirrors, and nothing warns
 * you. Measured: three projected the two subjects to 0.833 / 0.278 while the
 * derivation said 0.167 / 0.722, exact complements.
 *
 * Flipped here, in the one place the world is drawn, rather than negating the
 * derivation — the derivation is what the compiler and the backend share, and
 * it is the one that matches how a person describes a frame.
 */
const wx = (x: number) => -x
const vfov = (lens: number) => (2 * fov(lens, SENSOR_H) * 180) / Math.PI

/** A body: a box for the torso, a sphere for the head, a nub for the nose so
 *  facing is legible at a glance — the one thing a plan view needs and a
 *  featureless block cannot say. */
function Figure({ mark, on, onDown }: {
  mark: Mark; on: boolean; onDown: (e: ThreeEvent<PointerEvent>) => void
}) {
  return (
    <group position={[wx(mark.x), 0, mark.z]}
           rotation={[0, -(mark.yaw * Math.PI) / 180, 0]}
           onPointerDown={onDown}>
      <mesh position={[0, FIG_H * 0.36, 0]}>
        <boxGeometry args={[FIG_W, FIG_H * 0.72, FIG_W * 0.55]} />
        <meshStandardMaterial color={on ? '#e8e8e8' : GREY} roughness={1} />
      </mesh>
      <mesh position={[0, FIG_H * 0.85, 0]}>
        <sphereGeometry args={[0.13, 16, 12]} />
        <meshStandardMaterial color={on ? '#e8e8e8' : GREY} roughness={1} />
      </mesh>
      {/* which way they face */}
      <mesh position={[0, FIG_H * 0.85, 0.15]}>
        <sphereGeometry args={[0.045, 10, 8]} />
        <meshStandardMaterial color="#111" roughness={1} />
      </mesh>
    </group>
  )
}

/** The shot camera, drawn as a body with its lens cone — only ever seen from
 *  the plan, because from the frame you are standing in it. */
function CameraBody({ cam, path, onDown, onPath }: {
  cam: Cam; path: Path
  onDown: (e: ThreeEvent<PointerEvent>) => void
  onPath: (e: ThreeEvent<PointerEvent>) => void
}) {
  const half = fov(cam.lens, 36)
  const cone = useMemo(() => {
    const s = new THREE.Shape()
    s.moveTo(0, 0)
    s.lineTo(Math.sin(-half) * 14, Math.cos(-half) * 14)
    s.lineTo(Math.sin(half) * 14, Math.cos(half) * 14)
    return new THREE.ShapeGeometry(s)
  }, [half])
  return (
    <group position={[wx(cam.x), 0, cam.z]}
           rotation={[0, -(cam.yaw * Math.PI) / 180, 0]}>
      <mesh geometry={cone} position={[0, 0.02, 0]} rotation={[-Math.PI / 2, 0, 0]}>
        <meshBasicMaterial color="#fff" transparent opacity={0.06} side={THREE.DoubleSide} />
      </mesh>
      <mesh position={[0, cam.y, 0]} onPointerDown={onDown}>
        <boxGeometry args={[0.26, 0.2, 0.42]} />
        <meshStandardMaterial color="#f5f5f5" roughness={1} />
      </mesh>
      <mesh position={[0, cam.y / 2, 0]}>
        <cylinderGeometry args={[0.02, 0.02, cam.y, 6]} />
        <meshStandardMaterial color="#5a5a5a" roughness={1} />
      </mesh>
      {path && (
        <mesh position={[0, 0.05, 0]} onPointerDown={onPath} visible={false}>
          <sphereGeometry args={[0.3]} />
        </mesh>
      )}
    </group>
  )
}

/** Where the camera ends up, if it moves. Grabbable. */
function PathDot({ path, onDown }: {
  path: NonNullable<Path>; onDown: (e: ThreeEvent<PointerEvent>) => void
}) {
  return (
    <mesh position={[wx(path.x ?? 0), 0.06, path.z ?? 0]} onPointerDown={onDown}>
      <sphereGeometry args={[0.16, 14, 10]} />
      <meshStandardMaterial color="#fbbf24" roughness={1} />
    </mesh>
  )
}

/** Turns a pointer event into a point on the floor, which is the only mapping
 *  either view needs — every drag here is somebody moving something on the
 *  ground, never dragging a picture around. */
function Floor({ onDrag }: { onDrag: (x: number, z: number) => void }) {
  const { camera, raycaster, pointer } = useThree()
  const plane = useMemo(() => new THREE.Plane(new THREE.Vector3(0, 1, 0), 0), [])
  const hit = useRef(new THREE.Vector3())
  return (
    <mesh
      rotation={[-Math.PI / 2, 0, 0]} position={[0, -0.001, 6]}
      onPointerMove={() => {
        raycaster.setFromCamera(pointer, camera)
        if (raycaster.ray.intersectPlane(plane, hit.current)) {
          onDrag(wx(hit.current.x), hit.current.z)
        }
      }}
    >
      <planeGeometry args={[80, 80]} />
      <meshBasicMaterial visible={false} />
    </mesh>
  )
}

/**
 * The plan camera, aimed with `lookAt` rather than a hand-written rotation.
 *
 * Written by hand it pointed away from the set and rendered an empty floor
 * with the lens cone floating in it — a three.js camera looks down -Z, so the
 * Euler angles that *feel* like "above, looking down at the action" are not
 * the ones that are. Aiming at a point is the same instruction a human would
 * give, and it cannot be off by a sign.
 *
 * It also follows the action: the target is the mean of the marks, so adding
 * somebody or walking them away does not leave them off the edge of the plan.
 */
function PlanCamera({ focus }: { focus: [number, number, number] }) {
  const ref = useRef<THREE.PerspectiveCamera>(null)
  useFrame(() => {
    const c = ref.current
    if (!c) return
    c.position.set(focus[0], 12, focus[2] - 7)
    c.lookAt(focus[0], 0, focus[2])
  })
  return <PerspectiveCamera ref={ref} makeDefault fov={42} near={0.1} far={200} />
}

/** Dev only: publishes the live camera so the overlay can be checked against
 *  three's own projection rather than against my reading of a screenshot. */
function Probe3({ view }: { view: 'lens' | 'plan' }) {
  const { camera } = useThree()
  ;(window as unknown as Record<string, unknown>)[`__cam_${view}`] = camera
  return null
}

type Grab =
  | { t: 'mark'; i: number } | { t: 'markyaw'; i: number }
  | { t: 'cam' } | { t: 'camyaw' } | { t: 'path' } | null

export function Stage({
  cam, marks, path, grab, setGrab, onFloor, view,
}: {
  cam: Cam; marks: Mark[]; path: Path
  grab: Grab; setGrab: (g: Grab) => void
  onFloor: (x: number, z: number) => void
  view: 'lens' | 'plan'
}) {
  const lens = view === 'lens'
  const focus: [number, number, number] = marks.length
    ? [wx(marks.reduce((n, m) => n + m.x, 0) / marks.length), 0,
       marks.reduce((n, m) => n + m.z, 0) / marks.length]
    : [wx(cam.x), 0, cam.z + 3]
  return (
    <Canvas dpr={[1, 2]} style={{ background: '#0b0d10' }} shadows={false}>
      {lens ? (
        <PerspectiveCamera
          makeDefault fov={vfov(cam.lens)} position={[wx(cam.x), cam.y, cam.z]}
          rotation={[0, -(cam.yaw * Math.PI) / 180 + Math.PI, 0]} near={0.05} far={200}
        />
      ) : (
        <PlanCamera focus={focus} />
      )}
      <ambientLight intensity={1.2} />
      <directionalLight position={[4, 9, -3]} intensity={1.1} />
      <Grid args={[60, 60]} cellSize={1} sectionSize={5} infiniteGrid
            cellColor="#1b1f24" sectionColor="#2a3038" fadeDistance={38} />
      <Floor onDrag={onFloor} />
      <Probe3 view={view} />
      {marks.map((m, i) => (
        <Figure key={m.id} mark={m}
                on={!!grab && 'i' in grab && grab.i === i}
                onDown={(e) => { e.stopPropagation(); setGrab({ t: 'mark', i }) }} />
      ))}
      {!lens && (
        <>
          <CameraBody cam={cam} path={path}
                      onDown={(e) => { e.stopPropagation(); setGrab({ t: 'cam' }) }}
                      onPath={(e) => { e.stopPropagation(); setGrab({ t: 'path' }) }} />
          {path && <PathDot path={path}
                            onDown={(e) => { e.stopPropagation(); setGrab({ t: 'path' }) }} />}
        </>
      )}
    </Canvas>
  )
}

export type { Grab }
export { compile, see, norm }
