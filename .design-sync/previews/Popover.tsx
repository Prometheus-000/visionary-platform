import { useState } from 'react'
import { NumInput, Popover } from 'visionary-web'

/**
 * The card harness paints its page white; Visionary is a black-only system —
 * `ui.css` sets `body{background:#000;color:#f5f5f5}` and every component here
 * is drawn for that ground. The harness stylesheet loads after `styles.css`, so
 * the preview paints the ground itself rather than the shipped CSS fighting it
 * with `!important` — a real design gets the black from `body` the ordinary way.
 *
 * Normal flow, with a height, rather than `position:fixed` — the harness puts
 * `transform:translateZ(0)` on the cell, which makes it the containing block for
 * fixed descendants, so `inset:0` resolves against a box that has collapsed to
 * nothing. The same transform is what keeps the portalled overlay in the card.
 */
function Dark({ children }: { children: React.ReactNode }) {
  return (
    <div style={{
      background: 'var(--bg)', color: 'var(--fg)', minHeight: '100vh', padding: 20,
      font: '14px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", Inter, sans-serif',
    }}>{children}</div>
  )
}

function Row({ label, hint, children }: { label: string; hint?: string; children: React.ReactNode }) {
  return (
    <label className="frow">
      <span>{label}</span>
      {children}
      {hint && <i>{hint}</i>}
    </label>
  )
}

/**
 * `className` is the whole of the variation: the stylesheet selects on `menu`,
 * `menu form`, `menu sizer` and `pal`, and Popover itself only ever positions
 * and dismisses.
 */
function Anchored({ label, className, children }: {
  label: string; className: string; children: React.ReactNode
}) {
  const [anchor, setAnchor] = useState<HTMLElement | null>(null)
  return (
    <Dark>
      <button ref={setAnchor} className="opt" type="button">{label}</button>
      {anchor && <Popover anchor={anchor} className={className} onClose={() => {}}>{children}</Popover>}
    </Dark>
  )
}

/**
 * `menu form` — the Sampling popover, which is where the rarely-touched
 * controls went when the console was sorted by how often you reach for
 * something.
 */
export function FormVariant() {
  const [steps, setSteps] = useState('')
  const [cfg, setCfg] = useState('')
  return (
    <Anchored label="Krea 2 Turbo" className="menu form">
      <Row label="Sampler">
        <select defaultValue="euler"><option>euler</option><option>dpmpp_2m</option></select>
      </Row>
      <Row label="Scheduler">
        <select defaultValue="simple"><option>simple</option><option>karras</option></select>
      </Row>
      <Row label="Steps">
        <NumInput value={steps} onValue={setSteps} inputMode="numeric" base={8} placeholder="8" />
      </Row>
      <Row label="CFG">
        <NumInput value={cfg} onValue={setCfg} fine={0.1} bigStep={1} base={1} placeholder="1.0" />
      </Row>
    </Anchored>
  )
}

/** `menu` — a plain list of commands, the shape `Menu` builds on. */
export function MenuVariant() {
  return (
    <Anchored label="16:9 · 720p" className="menu">
      <button type="button">Copy prompt</button>
      <button type="button">Reuse settings</button>
      <hr />
      <button type="button" className="danger">Delete</button>
    </Anchored>
  )
}

/**
 * A long list, which is where the positioning earns itself.
 *
 * The popover is clamped to the viewport as well as anchored, and scrolls
 * inside its own `max-height` — a menu tall enough to need flipping above its
 * button is also tall enough for `top - h` to land off the top of the window,
 * and the rows past the edge would be unreachable, because the scrollbar is
 * inside the menu and nothing outside it scrolls them back. Neither the flip
 * nor the clamp is visible in a still, so this shows the list itself.
 */
export function ModelList() {
  return (
    <Anchored label="Model" className="menu">
      {['Krea 2 Turbo', 'Wan 2.2 TI2V 5B', 'Wan 2.2 A14B t2v',
        'MiniMax-H3 t2v', 'MiniMax-H3 ref2va',
      ].map((m) => <button key={m} type="button">{m}</button>)}
    </Anchored>
  )
}
