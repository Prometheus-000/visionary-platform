import { useState } from 'react'
import { NumInput } from 'visionary-web'

/**
 * The card harness paints its page white; Visionary is a black-only system —
 * `ui.css` sets `body{background:#000;color:#f5f5f5}` and every component here
 * is drawn for that ground. The harness stylesheet loads after `styles.css`, so
 * the preview paints the ground itself rather than the shipped CSS fighting it
 * with `!important` — a real design gets the black from `body` the ordinary way.
 */
function Dark({ children }: { children: React.ReactNode }) {
  return (
    <div style={{
      background: 'var(--bg)', color: 'var(--fg)', padding: 16, borderRadius: 8,
      font: '14px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", Inter, sans-serif',
    }}>{children}</div>
  )
}

/** The console's own form row: label, control, and an optional dim clause under it. */
function Row({ label, hint, children }: { label: string; hint?: string; children: React.ReactNode }) {
  return (
    <label className="frow">
      <span>{label}</span>
      {children}
      {hint && <i>{hint}</i>}
    </label>
  )
}

const SEED_HINT = 'Blank draws a new one. A seed worth keeping is on the render that used it.'

/**
 * The Sampling popover's form, which is where every one of these boxes lives.
 *
 * Each field carries the step it was given in the app: Steps walks by 1, CFG by
 * 0.1, Shift by 0.05 — a shift stepped by the default 8 would skip every value
 * the model accepts.
 */
export function SamplingFields() {
  const [steps, setSteps] = useState('')
  const [cfg, setCfg] = useState('')
  const [shift, setShift] = useState('1.15')
  const [seed, setSeed] = useState('')
  return (
    <Dark>
      <div className="menu form" style={{ position: 'static', maxWidth: 320 }}>
        <Row label="Steps">
          <NumInput value={steps} onValue={setSteps} inputMode="numeric" base={8} placeholder="8" />
        </Row>
        <Row label="CFG">
          <NumInput value={cfg} onValue={setCfg} fine={0.1} bigStep={1} base={1} placeholder="1.0" />
        </Row>
        <Row label="Shift">
          <NumInput value={shift} onValue={setShift} fine={0.05} bigStep={0.5} placeholder="1.15" />
        </Row>
        <Row label="Seed" hint={SEED_HINT}>
          <NumInput value={seed} onValue={setSeed} inputMode="numeric" placeholder="random" />
        </Row>
      </div>
    </Dark>
  )
}

/**
 * Width and Height, where the coarse step is the VAE's grid.
 *
 * Cmd-Up moves by 8 so it always lands on a size the model can render, rather
 * than one it will floor on the way through.
 */
export function DimensionBoxes() {
  const [w, setW] = useState('1152')
  const [h, setH] = useState('864')
  return (
    <Dark>
      <div className="menu form" style={{ position: 'static', maxWidth: 320 }}>
        <Row label="Width">
          <NumInput value={w} onValue={setW} fine={1} bigStep={8} inputMode="numeric" />
        </Row>
        <Row label="Height">
          <NumInput value={h} onValue={setH} fine={1} bigStep={8} inputMode="numeric" />
        </Row>
      </div>
    </Dark>
  )
}

/** An empty box counts from the checkpoint's own number, shown as the placeholder. */
export function EmptyCountsFromBase() {
  const [v, setV] = useState('')
  return (
    <Dark>
      <div className="menu form" style={{ position: 'static', maxWidth: 320 }}>
        <Row label="Steps" hint="Empty, so Up gives 9 — one more than the model would have used.">
          <NumInput value={v} onValue={setV} inputMode="numeric" base={8} placeholder="8" />
        </Row>
      </div>
    </Dark>
  )
}
