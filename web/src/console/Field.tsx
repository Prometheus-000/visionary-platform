import { useEffect, useLayoutEffect, useRef } from 'react'

import { IconPhoto, IconPlay } from '../icons'
import { caretProps } from '../lora/caret'
import { nudgeLora } from '../lora/tokens'
import { negAllowed, supports, useStore } from '../store'
import { autoGrow } from './fieldMax'
import { moveClause } from './moveClause'
import { Reroll } from './Reroll'
import { Rewrite } from './Rewrite'
import type { Marks } from './marks'
import { runs } from './useDocument'

/**
 * The prompt, its negative, and the chip that says what you are making.
 *
 * One prompt for both kinds. It is the same sentence either way, and losing it to a
 * mode switch is the fastest way to make two things that should feel like one feel
 * like two — so the switch is a chip *inside* the field rather than in the chrome:
 * which one you get is a property of what you are making, not an address you navigate
 * to.
 *
 * The strip goes in here too (`children`), which is what the vanilla page's `.bar2`
 * does. The row under the prompt was half empty — the Image/Video chip is 150px of a
 * 1456px line — while a whole separate strip sat beneath carrying everything else.
 * One row instead of two: the chip says what you are making and the rest of the line
 * says how, which is the same sentence.
 */
export function Field({
  consoleEl,
  onSubmit,
  children,
}: {
  consoleEl: React.RefObject<HTMLDivElement | null>
  onSubmit: () => void
  children: React.ReactNode
}) {
  const s = useStore()
  const prompt = useRef<HTMLTextAreaElement>(null)
  const neg = useRef<HTMLTextAreaElement>(null)
  const mirror = useRef<HTMLDivElement>(null)
  const ok = negAllowed(s)
  const live = s.negOn && ok ? neg : prompt
  // A ref rather than state: nothing renders differently while an IME is open,
  // and making this state would re-run the parse effect on every composition
  // start — which is the one moment it must not be disturbed.
  const composing = useRef(false)
  // **The automatic parse is off, and the marks with it.** It ran on every
  // pause, wrote its own prose into the box and underlined what it claimed to
  // have supplied. Measured over thirty blind render comparisons that document
  // never once beat the sentence it came from, and the provenance the marks were
  // drawn from was wrong every time the model asserted it — see
  // tools/parse-eval-2026-08-17. What replaced it is `Rewrite`: three jobs the
  // person asks for by name, which answers the same trust question by the
  // gesture rather than by colouring characters.
  //
  // `useDocument`, `marks.ts` and `Reroll` are still on disk and still wired to
  // `/api/parse`; nothing calls them from here. Left rather than deleted so the
  // measurement can be re-run against them, and so the deletion is one commit
  // somebody makes on purpose.
  const marks: Marks = { invented: [], spans: [] }

  // Switched away from under you: a model that reads no negative must not leave you
  // typing into a field it will not read. The text is kept — the next model may well
  // read it, and silently emptying a box someone wrote in is the one thing worse than
  // ignoring it. In an effect rather than inline, because this is a write and a render
  // that writes state is a render that can loop.
  useEffect(() => {
    if (s.negOn && !ok) s.setNegOn(false)
  }, [s.negOn, ok, s])

  useLayoutEffect(() => {
    autoGrow(live.current, consoleEl.current)
  }, [s.prompt, s.negative, s.negOn, live, consoleEl])

  // One placeholder, three states: it has to name the kind, and on the H3 checkpoints
  // it also has to ask for the soundtrack, which is denoised from the same sequence
  // and invented when you leave it out.
  const hint = s.kind === 'image'
    ? 'Describe an image…'
    : supports(s).audio
      ? 'Describe the shot, the motion — and the audio: dialogue, effects, music…'
      : 'Describe the shot and the motion…'

  /** Shared by both textareas: they are the same kind of comma-separated prose, and a
   *  chord that works in one box and not the one under it is a chord nobody trusts. */
  const keys = (
    value: string,
    write: (v: string) => void,
  ) => (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    const el = e.currentTarget
    if (e.altKey && !e.metaKey && !e.ctrlKey && !e.nativeEvent.isComposing
        && (e.key === 'ArrowLeft' || e.key === 'ArrowRight')) {
      const moved = moveClause(value, el.selectionStart ?? 0, e.key === 'ArrowRight' ? 1 : -1)
      // Falls through to the OS's word-jump when there is nowhere to go, which is the
      // honest answer to ⌥← on the first clause in the box.
      if (!moved) return
      e.preventDefault()
      write(moved.value)
      requestAnimationFrame(() => el.setSelectionRange(moved.caret, moved.caret))
      return
    }
    // ⌘↑ / ⌘↓ on a strength, the way the same chord nudges attention weights in
    // Automatic. Only swallowed when the caret was actually in a token; outside one,
    // ⌘↑ is still the caret-to-top the rest of the OS says it is.
    if ((e.metaKey || e.ctrlKey) && (e.key === 'ArrowUp' || e.key === 'ArrowDown')) {
      const w = nudgeLora(value, el.selectionStart ?? 0, e.key === 'ArrowUp' ? 0.05 : -0.05)
      if (!w) return
      e.preventDefault()
      write(w.value)
      requestAnimationFrame(() => el.setSelectionRange(w.caret, w.select))
      return
    }
    // ⌘Z while there is a parse write to take back. Native undo is already
    // superseded on three paths in this field — `moveClause`, `nudgeLora` and
    // the `+ LoRA` caret sink all write the value through React — so this is one
    // more, not a new kind of thing. It falls through to the browser's own undo
    // when the slot is empty, which is every keystroke that is not the one
    // immediately after a document landed.
    if ((e.metaKey || e.ctrlKey) && !e.shiftKey && e.key === 'z'
        && write === s.setPrompt && useStore.getState().docUndo) {
      e.preventDefault()
      s.undoDoc()
      return
    }
    // Shift+Enter keeps the newline, because prompts here are prose and paragraphs in
    // them are real. ⌘/Ctrl+Enter works too: it is what the muscle expects from every
    // other box that submits. isComposing, because an IME's Enter is committing a
    // character, not submitting.
    if (e.key === 'Enter' && !e.nativeEvent.isComposing && !e.shiftKey && !e.altKey) {
      e.preventDefault()
      onSubmit()
    }
  }

  return (
    <div className={['field', ok ? 'has-neg' : '', s.negOn && ok ? 'on-neg' : ''].filter(Boolean).join(' ')}>
      {/* The marks, painted on a copy of the prompt sitting exactly behind it.
          A textarea cannot style a range of its own value, and the alternative —
          contenteditable — would take the caret, the undo stack and every chord
          in `keys` with it. ⌥←/→, ⌘↑/↓, Enter-to-submit and the `+ LoRA` caret
          sink all keep working here because the thing you type into is still a
          textarea and nothing about it changed. */}
      <div className={`mk-mirror${s.negOn && ok ? ' hide' : ''}`} ref={mirror} aria-hidden="true">
        {runs(s.prompt, marks).map((r, i) => (
          <span key={i}
                className={[r.span ? 'mk-el' : '', r.invented ? 'mk-i' : '']
                  .filter(Boolean).join(' ') || undefined}>{r.text}</span>
        ))}
      </div>
      {/* Read-only while a rewrite is in flight, which is the whole of the
          staleness fix. The answer is written into this box when it lands, and
          it is the rewrite of the sentence that was here when the button was
          pressed — so typing during the wait means a stale answer arrives on
          top of newer words. A guard comparing the two was the other way to fix
          it and it is more machinery for a worse outcome: it would throw the
          answer away *after* paying for it. The rewrite is seconds, and a field
          that cannot drift is a field nothing has to be reconciled with.

          `readOnly` rather than `disabled`: disabled takes the caret and the
          selection with it, so the box loses your place for the length of a
          request you are about to get an answer to. */}
      <textarea id="prompt" ref={prompt} rows={1} placeholder={hint}
                className={s.negOn && ok ? 'hide' : ''}
                readOnly={!!s.rewriting}
                aria-busy={!!s.rewriting}
                value={s.prompt}
                onScroll={(e) => {
                  // The mirror has no scrollbar of its own — it is shorter than
                  // nothing and taller than the box only because the text is.
                  if (mirror.current) mirror.current.scrollTop = e.currentTarget.scrollTop
                }}
                onChange={(e) => s.setPrompt(e.target.value)}
                onKeyDown={keys(s.prompt, s.setPrompt)}
                // The two halves of one flag. `isComposing` is already this
                // codebase's idiom — the Enter guard in `keys` uses it — and
                // what is new here is only that a *timer* needs the state where
                // an event could carry it on itself.
                onCompositionStart={() => { composing.current = true }}
                onCompositionEnd={() => { composing.current = false }}
                {...caretProps('prompt', (v) => useStore.getState().setPrompt(v))} />
      {/* The same field in a different sign. Hidden outright on a model that reads no
          negative — H3 is guidance-distilled, Krea 2 Turbo runs at CFG 1.0, and on
          both of those a negative prompt is a promise the sampler will not keep. */}
      <textarea id="neg" ref={neg} rows={1} className={s.negOn && ok ? '' : 'hide'}
                placeholder="Negative prompt — what to steer away from"
                value={s.negative}
                onChange={(e) => s.setNegative(e.target.value)}
                onKeyDown={keys(s.negative, s.setNegative)} />
      {/* Rooted on the sentence rather than beside it, so there is nothing at
          rest and nothing to dismiss. */}
      {!(s.negOn && ok) && <Reroll mirror={mirror} el={prompt} />}
      {ok && (
        <button type="button" id="neg-toggle"
                className={`neg-t${s.negative.trim() ? ' filled' : ''}`}
                title="Which prompt you are writing. Click to switch; the negative is only read at CFG above 1."
                onClick={() => {
                  const to = !s.negOn
                  s.setNegOn(to)
                  requestAnimationFrame(() => (to ? neg : prompt).current?.focus())
                }}>
          {s.negOn ? 'negative' : 'positive'}
        </button>
      )}

      {/* Under the sentence and above the strip, because it acts on the sentence
          and is reached right after writing one. Empty until there is prose, so
          it costs the console nothing at rest. */}
      {!(s.negOn && ok) && <Rewrite />}

      <div className="bar2">
        {/* One icon, not a pair of chips. This is the rare case where a glyph alone
            carries it: a camera and a play triangle are not a vocabulary anyone has to
            learn, and the control has exactly two states so there is nothing a second
            chip could disambiguate. It shows the state you *are* in rather than the
            one you would get. */}
        <button className="opt ib" id="kind-toggle" type="button"
                title={s.kind === 'video' ? 'Video — tap for image' : 'Image — tap for video'}
                onClick={() => s.setKind(s.kind === 'video' ? 'image' : 'video')}>
          {s.kind === 'video' ? <IconPlay /> : <IconPhoto />}
        </button>
        {children}
      </div>
    </div>
  )
}
