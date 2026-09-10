import { useCallback, useState } from 'react'

import { Stage, type Grab } from './Stage'
import { compile, norm, see, type Cam, type Mark, type Path } from './derive'
import './blocking.css'

/**
 * The probe. Two views of one stage, and the sentence underneath.
 *
 * What it is testing is one thing: **does moving a body communicate intent
 * without a label?** So the prose is always on screen and always live — a
 * stage that does not show you the sentence it writes is a 3D toy, and a
 * sentence that does not change as you drag proves nothing about the gesture.
 */

const START = (): { cam: Cam; marks: Mark[] } => ({
  cam: { x: 0, z: 0, y: 1.5, yaw: 0, lens: 40 },
  marks: [
    { id: 1, x: -0.9, z: 3.0, yaw: 170, label: 'a woman in a red coat' },
    { id: 2, x: 0.9, z: 4.5, yaw: 200, label: 'a man at the counter' },
  ],
})

export function Probe() {
  const [{ cam, marks }, setWorld] = useState(START)
  const [path, setPath] = useState<Path>(null)
  const [secs, setSecs] = useState(6)
  const [grab, setGrab] = useState<Grab>(null)

  // Every drag is the same act — something moving on the floor — so there is
  // one handler and the grab says what. No modes, nothing to select first.
  const onFloor = useCallback((x: number, z: number) => {
    if (!grab) return
    setWorld((w) => {
      if (grab.t === 'mark') {
        const marks = w.marks.map((m, i) => (i === grab.i ? { ...m, x, z } : m))
        return { ...w, marks }
      }
      if (grab.t === 'markyaw') {
        const marks = w.marks.map((m, i) => i === grab.i
          ? { ...m, yaw: norm((Math.atan2(x - m.x, z - m.z) * 180) / Math.PI) } : m)
        return { ...w, marks }
      }
      if (grab.t === 'cam') return { ...w, cam: { ...w.cam, x, z } }
      if (grab.t === 'camyaw') {
        return { ...w, cam: { ...w.cam,
          yaw: norm((Math.atan2(x - w.cam.x, z - w.cam.z) * 180) / Math.PI) } }
      }
      return w
    })
    if (grab.t === 'path') setPath({ x, z, yaw: cam.yaw })
  }, [grab, cam.yaw])

  const { clauses, tail, boxes } = compile(cam, marks, path, secs)
  const set = (p: Partial<Cam>) => setWorld((w) => ({ ...w, cam: { ...w.cam, ...p } }))

  return (
    <div className="bp" onPointerUp={() => setGrab(null)}
         onPointerLeave={() => setGrab(null)}>
      <header>
        <h1>Blocking</h1>
        <span className="sub">marks on a floor &rarr; the sentence H3 reads</span>
      </header>

      <p className="hint">
        <b>Drag a body in either view.</b> On the plan you can also drag the
        camera and, with a move armed, where it ends up. Nothing is applied and
        nothing is confirmed &mdash; the prose below is what the compiler would
        write, right now.
      </p>

      <div className="views">
        <section>
          <h2>Through the lens <span>{cam.lens | 0}mm &middot; {cam.y.toFixed(2)}m</span></h2>
          <div className="vp">
            <Stage cam={cam} marks={marks} path={path} grab={grab}
                   setGrab={setGrab} onFloor={onFloor} view="lens" />
            {/* The projected boxes, over the render they describe — the same
                rectangles Krea 2 gets as regions. */}
            <svg viewBox="0 0 100 100" preserveAspectRatio="none" className="boxes">
              {boxes.map(({ m, b }) => b && (
                <rect key={m.id} x={b.x * 100} y={b.y * 100}
                      width={b.w * 100} height={b.h * 100} />
              ))}
            </svg>
          </div>
        </section>
        <section>
          <h2>The floor <span>plan</span></h2>
          <div className="vp">
            <Stage cam={cam} marks={marks} path={path} grab={grab}
                   setGrab={setGrab} onFloor={onFloor} view="plan" />
          </div>
        </section>
      </div>

      <div className="bar">
        <label>lens
          <input type="range" min={14} max={135} value={cam.lens}
                 onChange={(e) => set({ lens: +e.target.value })} /></label>
        <label>height
          <input type="range" min={20} max={300} value={cam.y * 100}
                 onChange={(e) => set({ y: +e.target.value / 100 })} /></label>
        <label>yaw
          <input type="range" min={-180} max={180} value={cam.yaw}
                 onChange={(e) => set({ yaw: +e.target.value })} /></label>
        <label>shot
          <input type="range" min={2} max={15} value={secs}
                 onChange={(e) => setSecs(+e.target.value)} /></label>
        <span className="tag">{secs}s</span>
        <button onClick={() => setWorld((w) => w.marks.length >= 8 ? w : {
          ...w, marks: [...w.marks, {
            id: Date.now(), x: (Math.random() - 0.5) * 3,
            z: 2 + Math.random() * 4, yaw: 180,
            label: `a body (${w.marks.length + 1})` }] })}>+ body</button>
        <button className={path ? 'on' : ''}
                onClick={() => setPath(path ? null : { x: cam.x, z: cam.z + 1.2, yaw: cam.yaw })}>
          camera move: {path ? 'drag the dot' : 'off'}</button>
        <button onClick={() => { setWorld(START()); setPath(null) }}>reset</button>
      </div>

      <div className="out">
        <h2>what the compiler writes</h2>
        <pre>
          <span className="dim">[Shot 1] </span>
          {clauses.length ? <b>{clauses.join(' ')} </b>
                          : <span className="dim">nobody in frame. </span>}
          {tail.join(' ')}
        </pre>
      </div>
      <div className="out">
        <h2>and the same arrangement as Krea 2 regions</h2>
        <pre>{boxes.filter((o) => o.b).length
          ? boxes.map(({ m, b }, i) => b
              ? `S${i + 1}  x=${b.x.toFixed(3)}  y=${b.y.toFixed(3)}  `
                + `w=${b.w.toFixed(3)}  h=${b.h.toFixed(3)}   ${m.label}\n`
              : '').join('')
          : 'nobody in frame — no boxes, which is the right answer rather than '
            + 'a zero-area one'}</pre>
      </div>
    </div>
  )
}

export { see }
