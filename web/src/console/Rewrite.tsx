import { failed } from '../api/client'
import { rewrite } from '../api/routes'
import { stripLoras } from '../lora/tokens'
import { useStore } from '../store'

/**
 * The third door into the prompt, beside `+ LoRA` and `Shot`.
 *
 * **In the strip, not on a row of its own.** It shipped as `#rewrite-rail` — a
 * flex row under the textarea holding one button — which is the exact price
 * `#shot-rail:empty` was written to refuse: a line per button, 34px at rest,
 * forever, out of a console capped at 30% of the viewport. Shot paid that once
 * and came back into the strip for it. This did not need to learn it twice.
 *
 * So it is shaped like its neighbours rather than like itself. `+ LoRA` writes a
 * token at the caret, `Shot` opens a vocabulary, this rewrites the sentence —
 * three doors onto the same box, and they read as three of a kind because they
 * are built as three of a kind. Same `.s`, same height, same spacing.
 *
 * **Disabled rather than absent when there is nothing to act on.** Hiding it
 * would reflow the strip on the first keystroke and move `+ LoRA` under the
 * pointer on its way to being pressed. A door that is visibly shut is also the
 * only version of this that says the feature exists before you have typed.
 */
export function Rewrite({ onNote }: {
  /** Where a press that wrote nothing gets to say so — the console's reserved
   *  note line. Passed in rather than rendered here because this component lives
   *  inside the strip and that line is a row below it; see `Console`.
   *
   *  `calm` because that line is amber by default and only ever says what is
   *  wrong. "Nothing to change" is the one thing said here that is not a
   *  complaint, and painting it the same colour as a dead call would undo the
   *  distinction this note exists to draw. */
  onNote: (text: string, calm?: boolean) => void
}) {
  const s = useStore()
  const ops = s.state?.rewrite_ops ?? []
  const prose = stripLoras(s.prompt).trim()
  if (!ops.length) return null

  const run = async (op: string, label: string) => {
    if (s.rewriting) return
    s.setRewriting(op)
    // The previous press's note goes with the new press. Left up, a stale
    // "came back empty" would sit beside a run that is succeeding.
    onNote('')
    try {
      const r = await rewrite({ prose, op, kind: s.kind })
      // `failed` is the transport; `ok` is the route's own verdict. The route
      // answers with the original text on either, so the worst case here is the
      // box unchanged rather than the box emptied.
      //
      // **But the box unchanged is what every other outcome also looks like**,
      // and that is the fault this branch used to have all to itself: observed
      // live, `/api/rewrite` answered 200 with no `text` field and the page did
      // nothing at all — press, flash, nothing — which reads exactly like "your
      // prompt was already good". Both endings below say which one happened.
      // Neither of them touches the words; only the note line moves.
      if (!failed(r) && r.ok && r.text) {
        // Read fresh and compared on the same string `applyRewrite` compares, so
        // the note and the write can never disagree about whether anything moved.
        //
        // Nothing is lifted off and re-appended any more. A LoRA used to be
        // `<lora:…>` in this very field, so a rewrite that replaced the box
        // deleted somebody's stack — silently, because nothing on the page warns
        // about a token that was there a moment ago. It is a chip now and this
        // write cannot reach it.
        const before = useStore.getState().prompt
        s.applyRewrite(r.text)
        onNote(r.text === before
          ? `${label} found nothing to change — your prompt already reads as specific.`
          : '', true)
        return
      }
      onNote(failed(r)
        ? r.error
        : (r.error
           || `${label} came back with nothing — your prompt is untouched. Press it again.`))
    } finally {
      s.setRewriting(null)
    }
  }

  // `o.note`, served. It used to read "the way Krea 2 reads best" while MiniMax-H3 was
  // the model on screen — `rewrite_ops` comes off `/api/state`, fetched once at load
  // with no idea which kind you are on, so one string is read by both strips and a model
  // name in it is wrong in half the places the button appears. That was fixed where the
  // string is written (`REWRITE_OPS` in app.py) rather than papered over here, so this
  // stays a straight read: copy has one home, and a client-side override would be a
  // second one to keep in step.
  return (
    <>
      {ops.map((o) => (
        <button key={o.key} className="s" type="button"
                style={{ height: 32, padding: '0 10px' }}
                disabled={!prose || !!s.rewriting}
                aria-busy={s.rewriting === o.key}
                title={o.note}
                onClick={() => void run(o.key, o.label)}>
          {s.rewriting === o.key ? `${o.label}…` : o.label}
        </button>
      ))}
      {/* Only while there is something to take back, and it says what it
          reverses rather than naming the chord — the write is the surprising
          thing, not the shortcut. */}
      {s.docUndo && !s.rewriting && (
        <button className="s" type="button"
                style={{ height: 32, padding: '0 10px' }}
                title="Put back what you wrote (⌘Z in the prompt)"
                onClick={() => s.undoDoc()}>
          Undo
        </button>
      )}
    </>
  )
}
