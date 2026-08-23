import { useEffect, useLayoutEffect, useRef } from 'react'

import { caretProps } from '../lora/caret'
import { negAllowed, useStore } from '../store'
import { Doors } from '../scene/Doors'
import { Shots } from '../scene/Shots'
import { Duration } from './Duration'
import { growField } from './fieldMax'
import { moveClause } from './moveClause'

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
  onPalette,
  children,
}: {
  consoleEl: React.RefObject<HTMLDivElement | null>
  onSubmit: () => void
  /** The shot palette, opened from the video side's field-edge door. One popover
   *  shared by both strips — see `Doors`. */
  onPalette: (e: React.MouseEvent<HTMLElement>) => void
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
  // Switched away from under you: a model that reads no negative must not leave you
  // typing into a field it will not read. The text is kept — the next model may well
  // read it, and silently emptying a box someone wrote in is the one thing worse than
  // ignoring it. In an effect rather than inline, because this is a write and a render
  // that writes state is a render that can loop.
  useEffect(() => {
    if (s.negOn && !ok) s.setNegOn(false)
  }, [s.negOn, ok, s])

  useLayoutEffect(() => {
    growField(consoleEl.current)
  }, [s.prompt, s.negative, s.negOn, s.scene.shots, live, consoleEl])

  // The video side's two placeholders moved onto the row that shows them — it
  // has to name the kind, and on the H3 checkpoints it also has to ask for the
  // soundtrack, which is denoised from the same sequence and invented when you
  // leave it out. See `Shots`.
  const hint = 'Describe an image…'

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
      {/* **The prose is a timeline on one side and a sentence on the other.**
          H3 reads a document with named fields — shots, cut times, speaker IDs —
          and a text field has no shots in it, so the video side's box is the
          first row of `Shots` rather than this one. With one shot it renders the
          same `#prompt` under the same rules, which is the degrade the whole
          composer rests on: nothing changes until you ask for a second shot.

          The image side keeps this box. Krea 2 has no fields to fill in, so a
          timeline there would be a control the compiler discards. */}
      {s.kind === 'video' ? <Shots consoleEl={consoleEl} hide={s.negOn && ok}
                                   onSubmit={onSubmit} /> : (
      <>
      {/* **The mirror is the text you read**, and the textarea over it is
          transparent with only its caret coloured. It was built to paint marks
          onto the prompt — a textarea cannot style a range of its own value,
          and contenteditable would have taken the caret, the undo stack and
          every chord in `keys` with it. The marks are gone here and the
          arrangement stays, because unwinding it buys nothing — and the video
          side's mirror is painting mentions again, which is what it was for. */}
      <div className={`mk-mirror${s.negOn && ok ? ' hide' : ''}`} ref={mirror} aria-hidden="true">
        {/* Two spans, and the empty one is load-bearing: a mirror that ends
            exactly at its last character loses the newline just typed, and the
            copy behind the box stops matching the box by one line. */}
        <span>{s.prompt}</span>
        <span />
      </div>
      <textarea id="prompt" ref={prompt} rows={1} placeholder={hint}
                className={s.negOn && ok ? 'hide' : ''}
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
      </>
      )}
      {/* The same field in a different sign. Hidden outright on a model that reads no
          negative — H3 is guidance-distilled, Krea 2 Turbo runs at CFG 1.0, and on
          both of those a negative prompt is a promise the sampler will not keep. */}
      <textarea id="neg" ref={neg} rows={1} className={s.negOn && ok ? '' : 'hide'}
                placeholder="Negative prompt — what to steer away from"
                value={s.negative}
                onChange={(e) => s.setNegative(e.target.value)}
                onKeyDown={keys(s.negative, s.setNegative)} />
      {/* **The trailing edge carries composition, and the corner is shared.**
          `.neg-t` has had this corner to itself since it took it from the resize
          grip, and on Wan both are present — so they sit in one row rather than on
          top of each other. On H3 there is no negative branch at all and the doors
          have the corner; on the image side it is `.neg-t` alone, exactly as it
          was. */}
      <span className="fedge-row">
      {s.kind === 'video' && <Doors onPalette={onPalette} />}
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
      </span>

      {/* Under the sentence and above the strip, because it acts on the sentence
          and is reached right after writing one. Empty until there is prose, so
          it costs the console nothing at rest. */}

      <div className="bar2">
        {/* Leftmost, and it decides what everything to its right means. See
            `Duration` for why the image/video chip is gone rather than relabelled. */}
        <Duration />
        {children}
      </div>
    </div>
  )
}
