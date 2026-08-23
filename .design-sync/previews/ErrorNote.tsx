import { ErrorNote } from 'visionary-web'

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

/**
 * A sentence the page raised itself, with no server behind it.
 *
 * Several call sites do this — the message never had an `ApiError` shape and
 * should not have to be dressed up as one to render.
 */
export function BareMessage() {
  return <Dark><ErrorNote err="Give the set a name to save it." /></Dark>
}

/**
 * The split form: the sentence a person reads, and the server's own report
 * folded away behind it.
 *
 * Closed by default, so the traceback costs a line of chrome and no attention.
 * Deleting it instead was the easy version of this fix and the wrong one — it
 * is the only thing in the failure that says what actually broke.
 */
export function WithServerDetail() {
  return (
    <Dark>
      <ErrorNote
        err={{
          error: 'That checkpoint is not on the volume yet.',
          detail: [
            'FileNotFoundError: /workspace/models/krea2_turbo_fp8.safetensors',
            '  resolved volume: visionary',
            '  on the volume:   krea2_base_fp8.safetensors, wan2.2_ti2v_5B.safetensors',
            '  at app.py:2841 in _require_models',
          ].join('\n'),
        }}
      />
    </Dark>
  )
}

/** Two failures stacked, which is how they arrive in Settings. */
export function Stacked() {
  return (
    <Dark>
      <ErrorNote err="No LoRA named high — two files match. Qualify it by folder." />
      <ErrorNote
        style={{ marginTop: 10 }}
        err={{
          error: 'The HuggingFace token was refused.',
          detail: '401 Unauthorized — repo MiniMaxAI/MiniMax-H3 is gated.',
        }}
      />
    </Dark>
  )
}

/** Falsy renders nothing at all, so a call site needs no conditional of its own. */
export function NothingWhenEmpty() {
  return (
    <Dark>
      <ErrorNote err={null} />
      <span style={{ color: 'var(--dim)', fontSize: 12 }}>
        err is null — the component rendered nothing
      </span>
    </Dark>
  )
}
