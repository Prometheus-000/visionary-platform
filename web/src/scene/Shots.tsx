import { useLayoutEffect, useRef, useState } from 'react'

import { caretProps } from '../lora/caret'
import { moveClause } from '../console/moveClause'
import { growRows } from '../console/fieldMax'
import { resolveVid } from '../console/resolve'
import { supports, useStore } from '../store'
import { MentionMenu, complete, mentionAt, type Mention } from './Mentions'
import { Timeline } from './Timeline'
import { handleOf, sceneSeconds, times, type Shot } from './model'

/**
 * The timeline, where the video side's prompt box used to be.
 *
 * **It is always drawn as a timeline, including at one shot.** It was not: with
 * one row the strip and the gutter were suppressed so the surface was
 * byte-for-byte the prompt box it replaced, on the argument that a feature should
 * not announce itself before it is being used. The owner's reading of that,
 * verbatim: *"I don't even see a timeline."* Which is the same fault the icon
 * rule already names one layer down — a surface indistinguishable from the one it
 * replaced is a capability nobody can find, and the degrade it was protecting is
 * a fact about the *compiler*, not about what the page should look like. One shot
 * still compiles to the typed text byte-for-byte; it simply no longer pretends to
 * be a text box while doing it.
 *
 * **Each row fits its own content, up to three lines** — the budget arithmetic
 * that used to divide one allowance between them retired with the 30% number;
 * see `fieldMax.ts` for the ruling. A tall scene is an honest cost the console's
 * own overflow cap absorbs, not a mutual squeeze between the rows.
 */
export function Shots({ hide, onSubmit }: {
  /** The negative box is showing instead. Hidden rather than unmounted, so the
   *  caret, the scroll position and the selection are where you left them when
   *  you switch back — the same reason `#prompt` was only ever `.hide`d. */
  hide: boolean
  onSubmit: () => void
}) {
  const s = useStore()
  const box = useRef<HTMLDivElement>(null)
  const shots = s.scene.shots
  // A string on the composer, because an empty box means "the model's default"
  // — see `ResolvedVid`. The clock needs a number, and the fallback is one
  // second so a model with no length yet still divides rather than dividing by
  // zero; nothing is shown at that point anyway.
  //
  // **The track's total, not the menu's.** `times` normalises the shares against
  // whatever it is handed, so asking the menu put the gutter and the document on
  // different clocks the moment the bars outgrew it: three 4s shots read `05.33`
  // under an 8s menu where the document said `At 00:08.000`, with the ruler two
  // pixels away reading 12s and correct. Same rule `sceneSeconds` and `Timeline`
  // already state — the timeline is the clip's length — and this was the one
  // place still asking. Live, it makes `times` an identity over the bars, which
  // is exactly the arithmetic `_compile_h3_scene` does with the same number.
  const secs = sceneSeconds(s.scene) ?? (Number(resolveVid(s).seconds) || 1)
  const cuts = times(shots, secs)

  useLayoutEffect(() => {
    growRows(box.current)
  })

  // The selected shot is the one you are writing. One field, not a field per
  // shot: with time on its own axis the rows were carrying two jobs — where a
  // shot sits in the film, and what it says — and only the second one needs a
  // textarea. Falls back to the first, because `shotSel` can name a row that a
  // ⌫ removed.
  const sel = shots.find((x) => x.id === s.shotSel) ?? shots[0]
  const at = sel ? shots.indexOf(sel) : 0

  return (
    <div className={`tline${hide ? ' hide' : ''}`} ref={box}>
      <Timeline />
      {sel && (
        <Row key={sel.id} shot={sel} n={at} at={cuts[at]?.[0] ?? 0}
             onSubmit={onSubmit} />
      )}
    </div>
  )
}


/** `MM:SS.mmm` is the cut format the document takes; the gutter is a readout for
 *  a person, so it is the same instant at the precision a person reads. */
const tick = (t: number) => {
  const m = Math.floor(t / 60)
  const rest = (t - m * 60).toFixed(2).padStart(5, '0')
  return m ? `${String(m)}:${rest}` : rest
}

function Row({ shot, n, at, onSubmit }: {
  shot: Shot; n: number; at: number; onSubmit: () => void
}) {
  const s = useStore()
  const area = useRef<HTMLTextAreaElement>(null)
  const mirror = useRef<HTMLDivElement>(null)
  const caretEl = useRef<HTMLSpanElement>(null)
  // Where the caret is, and whether this row has it. State rather than a ref
  // because the mention menu renders from it — this is the one place in the app
  // where a caret position is not purely a handle on a DOM node.
  const [caret, setCaret] = useState(-1)
  // The mention already settled, by index of its `@`. Picking leaves the caret
  // inside the handle it just wrote, so without this the menu reopens onto the
  // name you have chosen and sits there — a picker that will not take yes for an
  // answer. Cleared on the next keystroke, because editing a handle is exactly
  // when the list should come back.
  const settled = useRef<number | null>(null)
  // **Always `#prompt`.** Everything that reaches for the prompt by id — the
  // stray-key focus in App.tsx, the checks, the Enter binding — is asking for
  // "the box you write in", and with one field per selection that is this one,
  // whichever shot is selected.
  const id = 'prompt'
  const write = (line: string) => { s.patchShot(shot.id, { line }) }

  const found = caret < 0 ? null : mentionAt(shot.line, caret)
  const mention: Mention | null = found && settled.current === found.at ? null : found

  /** Settle the mention on a handle and put the caret after it. */
  const pick = (handle: string) => {
    if (!mention) return
    const w = complete(shot.line, mention, handle)
    write(w.value)
    settled.current = mention.at
    setCaret(w.caret)
    // After the commit, for the reason `applyWrite` waits: the field is controlled
    // and a range set now would range over text React has not painted yet.
    requestAnimationFrame(() => {
      const el = area.current
      el?.focus()
      el?.setSelectionRange(w.caret, w.caret)
    })
  }

  const hint = n > 0
    ? 'What happens next…'
    : supports(s).audio
      ? 'Describe the shot, the motion — and the audio: dialogue, effects, music…'
      : 'Describe the shot and the motion…'

  const keys = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    const el = e.currentTarget
    if (e.altKey && !e.metaKey && !e.ctrlKey && !e.nativeEvent.isComposing
        && (e.key === 'ArrowLeft' || e.key === 'ArrowRight')) {
      const moved = moveClause(shot.line, el.selectionStart ?? 0,
                               e.key === 'ArrowRight' ? 1 : -1)
      if (!moved) return
      e.preventDefault()
      write(moved.value)
      requestAnimationFrame(() => { el.setSelectionRange(moved.caret, moved.caret) })
      return
    }
    // ⌫ on an empty row past the first removes it, which is the only way back
    // out of a timeline that does not need a control of its own. The first row
    // is never removable — a scene with no shots is a scene with nowhere to
    // type, and `_validate_scene` reads no shots as no scene at all.
    if (e.key === 'Backspace' && n > 0 && !shot.line && !shot.say.text) {
      e.preventDefault()
      s.dropShot(shot.id)
      return
    }
    if (e.key === 'Enter' && !e.nativeEvent.isComposing && !e.shiftKey && !e.altKey) {
      e.preventDefault()
      // ⌘⏎ submits from anywhere; a bare ⏎ in a row that already has something
      // in it starts the next shot, which is the gesture a timeline makes
      // available and a prompt box cannot.
      if (e.metaKey || e.ctrlKey || !shot.line.trim()) { onSubmit(); return }
      // **Split at the caret rather than append an empty row.** H3 timestamps a
      // shot boundary and nothing else — there is no timestamp for an action or
      // a hold — so the cut is the only timing control the model has, and
      // reaching the moment mid-sentence has to carry the tail across rather
      // than strand it. A caret at the end leaves nothing to carry, which is the
      // old behaviour exactly.
      s.splitShot(shot.id, el.selectionStart ?? shot.line.length)
      // **`#prompt`, like every other focus call in the app.** This reached for
      // `shot-${id}`, which has never been an element here: the row is remounted
      // by its key when the selection moves, so focus fell to <body> and nothing
      // put it back — and the next keystroke went to the stray-key handler in
      // `App`, which wrote it over shot 1. One dead selector, and the visible
      // failure was losing the sentence you had just written.
      requestAnimationFrame(() => {
        const box = document.getElementById('prompt') as HTMLTextAreaElement | null
        box?.focus()
        // At the end of what moved, which is where you were writing. A caret
        // left at 0 would put the next word in front of the tail.
        box?.setSelectionRange(box.value.length, box.value.length)
      })
    }
  }

  // The caret sink and this row want the same five events, and both have to run.
  // Spreading `caretProps` and then declaring one of its names above it silently
  // drops that handler — a later prop wins — which would take ⌥←/→ with it. So the
  // sink is built once and each override calls through to it.
  const sink = caretProps('prompt', write)
  /** Five events, for the reason `caretProps` needs five: focus alone misses a
   *  click that only moves the caret, and keyup alone misses a mouse selection. */
  const track = (e: React.SyntheticEvent<HTMLTextAreaElement>) => {
    setCaret(e.currentTarget.selectionStart ?? -1)
  }
  const also = (
    key: 'onKeyUp' | 'onClick' | 'onSelect' | 'onFocus',
    extra?: () => void,
  ) => (e: React.SyntheticEvent<HTMLTextAreaElement>) => {
    sink[key](e)
    extra?.()
    track(e)
  }

  return (
    <div className={`trow${shot.id === s.shotSel ? ' sel' : ''}`}>
      <span className="tnum" aria-hidden="true">
        {n + 1}<em>{tick(at)}</em>
      </span>
      <div className="tbox">
        <div className="mk-mirror" ref={mirror} aria-hidden="true">
          {/* The mirror is a glyph-for-glyph copy of the textarea, so a zero-width
              span inside it sits exactly where the caret sits — which is the only
              way anything on this page can know that. Emitted only while a mention
              is open, so the copy behind the box is otherwise untouched. */}
          <Painted line={shot.line} mark={mention?.at ?? null} markRef={caretEl} />
          {/* Load-bearing empty span — a mirror that ends exactly at its last
              character loses the newline just typed, and the copy behind the box
              stops matching the box by one line. */}
          <span />
        </div>
        <textarea id={id} ref={area} rows={1} placeholder={hint} value={shot.line}
                  onScroll={(e) => {
                    if (mirror.current) mirror.current.scrollTop = e.currentTarget.scrollTop
                  }}
                  onChange={(e) => { settled.current = null; write(e.target.value); track(e) }}
                  onKeyDown={keys}
                  onKeyUp={also('onKeyUp')}
                  onClick={also('onClick')}
                  onSelect={also('onSelect')}
                  onFocus={also('onFocus', () => { s.selectShot(shot.id) })}
                  // Closed on blur rather than left standing: the menu is about
                  // where the caret is, and a caret that has left the box is not
                  // anywhere. `-1` rather than `null` so `mentionAt` is never asked
                  // about a position that does not exist.
                  onBlur={() => { setCaret(-1) }} />
        {mention && (
          <MentionMenu anchorRef={caretEl} mention={mention} onPick={pick}
                       onClose={() => { setCaret(-1) }} />
        )}
      </div>
    </div>
  )
}

/**
 * The line with its mentions marked.
 *
 * A mention is stored as the literal text `@ava` and *painted* as a chip, which
 * is the whole reason the mirror survived the deletion of the marks it was built
 * for. The consequence is worth stating rather than discovering: edit the handle
 * and it stops being a mention — the words turn plain and the shot no longer
 * claims that subject.
 *
 * A handle nobody defined is marked as missing rather than left plain, because
 * the failure it prevents reads as the model ignoring you: `@ava` compiles to
 * those literal characters, which the encoder renders as nothing at all.
 */
function Painted({ line, mark, markRef }: {
  line: string
  /** Index of the `@` a mention menu is open on, or null. */
  mark: number | null
  markRef: React.RefObject<HTMLSpanElement | null>
}) {
  const cast = useStore((st) => st.scene.cast)
  const known = new Set(cast.map((c) => handleOf(c.name)).filter(Boolean))
  const out: React.ReactNode[] = []
  const re = /@([a-z0-9_]+)/gi
  let at = 0
  /** Everything up to `to`, with the caret marker spliced in if it falls inside. */
  const plain = (to: number) => {
    if (mark === null || mark < at || mark >= to) {
      if (to > at) out.push(line.slice(at, to))
    } else {
      if (mark > at) out.push(line.slice(at, mark))
      out.push(<span key="mk" className="tcaret" ref={markRef} />)
      if (to > mark) out.push(line.slice(mark, to))
    }
    at = to
  }
  for (const m of line.matchAll(re)) {
    const i = m.index
    plain(i)
    // A mention being typed carries the marker at its own `@`, which is where the
    // menu wants to hang — left-aligned with the name rather than with the caret,
    // so the list does not walk sideways as you type into it.
    const isMark = mark === i
    out.push(
      <span key={i} className={`men${known.has(m[1]!.toLowerCase()) ? '' : ' miss'}`}
            ref={isMark ? markRef : undefined}>
        {m[0]}
      </span>,
    )
    at = i + m[0].length
  }
  plain(line.length)
  return <>{out}</>
}
