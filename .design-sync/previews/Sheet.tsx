import { Sheet } from 'visionary-web'

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
      background: 'var(--bg)', color: 'var(--fg)', minHeight: '100vh', margin: -12, padding: 20,
      font: '14px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", Inter, sans-serif',
    }}>
      {children}
    </div>
  )
}

/**
 * The confirm that stands in for an undo.
 *
 * Deletion here unlinks, so the dialog is the safety net — which means it has
 * to say what is going and how much of it. A dialog that undersells the blast
 * radius is the failure mode this replaced.
 */
export function StopTrainingConfirm() {
  return (
    <Dark><Sheet id="stop-ask" onClose={() => {}}>
      <div className="sheet-head">
        <div>
          <h3 style={{ margin: 0 }}>Cancel k3nan-v3?</h3>
          <p className="sub" style={{ marginTop: 8, marginBottom: 0 }}>
            The GPU stops either way. Every epoch saved so far stays in
            <code> loras/</code> as a checkpoint — the LoRA as it was at that
            epoch, not the finished one — so a session stopped part-way is still
            the epochs it got through.
          </p>
        </div>
      </div>
      <div className="sess-acts" style={{ marginTop: 18 }}>
        <button className="t" type="button">Keep training</button>
        <button className="b" type="button">Stop</button>
      </div>
    </Sheet></Dark>
  )
}

/**
 * The metadata sheet — what a render was actually told.
 *
 * The typed prompt is shown rather than the compiled one: intent is the durable
 * half, and the compiled prompt is a receipt for whichever encoder was being
 * fed that day.
 */
export function MetadataSheet() {
  const rows: [string, string][] = [
    ['Model', 'Krea 2 Turbo'],
    ['Size', '1152 × 864 · 4:3'],
    ['Steps', '8'],
    ['CFG', '1.0'],
    ['Shift', '1.15'],
    ['Seed', '744012839'],
    ['LoRAs', '<lora:k3nan:1>'],
  ]
  return (
    <Dark><Sheet onClose={() => {}}>
      <div className="sheet-head">
        <div>
          <h3 style={{ margin: 0 }}>Details</h3>
          <p className="sub" style={{ marginTop: 8, marginBottom: 0 }}>
            empty diner, 3am — the light only from the sign outside
          </p>
        </div>
      </div>
      <dl style={{ display: 'grid', gridTemplateColumns: 'auto 1fr', gap: '8px 18px', margin: 0 }}>
        {rows.map(([k, v]) => (
          <div key={k} style={{ display: 'contents' }}>
            <dt style={{ color: 'var(--mut)', fontSize: 12 }}>{k}</dt>
            <dd style={{ margin: 0, fontSize: 12 }}>{v}</dd>
          </div>
        ))}
      </dl>
    </Sheet></Dark>
  )
}
