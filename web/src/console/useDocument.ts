import { useEffect, useLayoutEffect, useRef, useState } from 'react'

import { failed } from '../api/client'
import { parse } from '../api/routes'
import { stripLoras } from '../lora/tokens'
import { useStore } from '../store'
import { documentMarks, editDocument, remap, remapCaret,
         type Mark, type Marks } from './marks'

/** How long a pause counts as one, matching the disclosure under the rail. */
const PAUSE_MS = 500

const NONE: Marks = { invented: [], spans: [] }

/**
 * The model's reading of the prompt, kept true across typing.
 *
 * Two sources, and they are not interchangeable. The **parse** decides what was
 * invented, on a pause, because only the model knows. Every **edit** in between
 * carries the existing marks forward locally, because the model does not get
 * asked again until you stop — and if it did, it could not answer: it is handed
 * one string and cannot see which half of it was its own suggestion.
 *
 * A parse in flight is not allowed to land on text it did not read. That is the
 * whole of the guard below and it is the failure this would otherwise have: you
 * keep typing while the request is out, the reply arrives describing the
 * sentence from half a second ago, and the underlines settle over whatever
 * happens to be at those offsets now. The same guard gates the document — it
 * was always the right shape, it just gates one more write now.
 *
 * **The interpreter is shown prose with the LoRA syntax stripped.**
 * `<lora:k3nan:1>` is notation, not a sentence, and asking a model to structure
 * it is asking it to structure the app; the document is therefore keyed to the
 * stripped string, which is also exactly what `/api/generate` receives. Marks
 * are still placed against the *raw* box, because that is what the mirror
 * paints — and a clause a token sits inside of simply goes unmarked, under
 * `documentMarks`' own rule that a run it cannot find is skipped rather than
 * guessed at.
 */
export function useDocument(
  prompt: string,
  el: React.RefObject<HTMLTextAreaElement | null>,
  composing: React.RefObject<boolean>,
): Marks {
  const [marks, setMarks] = useState<Marks>(NONE)
  const seen = useRef(prompt)
  // Subscribed, not read once. The document is replaced by three things — a
  // parse landing, a reroll committing, a gallery card being reused — and marks
  // recomputed at only one of those are marks that are wrong after the other
  // two. Watching the document makes the paint a consequence of the document
  // rather than something each writer has to remember to do.
  const doc = useStore((st) => st.doc)
  /** Where the caret goes once React has actually written the new value. */
  const caretTo = useRef<number | null>(null)

  // Synchronous with the change, and watching the *value* rather than wrapping
  // a setter: `+ LoRA`, Reuse off a gallery card and the clause chords all write
  // the prompt through paths of their own, and marks that only tracked one of
  // them would be marks that lie after using any of the others.
  useLayoutEffect(() => {
    // **`was` is a local, and that is the whole of it.** A functional update is
    // lazy — React calls the updater during the *next* render, by which time
    // the assignment below has already moved `seen.current` to `prompt`. Read
    // through the ref, the updater therefore remapped the string onto itself,
    // `remap` returned early on `before === after`, and no mark has ever
    // survived an edit: the stale underline simply sat over whatever characters
    // now occupied its offsets until the next parse replaced it. Which is the
    // exact failure this feature exists to avoid — a mark that looks right,
    // over words nobody chose.
    const was = seen.current
    if (was !== prompt) {
      seen.current = prompt
      // The document carried across the same edit, so that touching one grey
      // run does not send the whole sentence back to be read again — see
      // `editDocument`. A patch keeps `doc.for` matching the box, which is what
      // stops the effect below re-parsing at all; anything it cannot patch is
      // left stale and the next pause replaces it.
      const st = useStore.getState()
      if (st.doc && st.doc.for === stripLoras(was)) {
        const next = editDocument(st.doc.elements, was, prompt)
        const prose = stripLoras(prompt)
        st.setDoc(next && next.length
          ? { for: prose, from: st.doc.from, elements: next, text: prose } : null)
      }
      // Both channels, carried by the same edit. They are two views of one
      // document and letting them drift would put a grey run outside the
      // underline it belongs to.
      setMarks((m) => ({
        invented: remap(m.invented, was, prompt),
        spans: remap(m.spans, was, prompt),
      }))
    }
    // **After the commit, not after a frame.** `requestAnimationFrame` can fire
    // before React has written the new value, and a range set on the old string
    // is then overwritten by the value change — which sends the caret to the
    // end of the prompt, the exact failure `remapCaret` exists to prevent. A
    // layout effect runs after the DOM has the new text, which is the only
    // moment this is safe.
    if (caretTo.current != null) {
      const to = Math.max(0, Math.min(prompt.length, caretTo.current))
      caretTo.current = null
      el.current?.setSelectionRange(to, to)
    }
  }, [prompt, el])

  // The document, painted. Whenever it describes exactly what is in the box its
  // marks are the truth; `remap` above is only the interim, carrying the last
  // ones across keystrokes until the next document arrives.
  useLayoutEffect(() => {
    if (doc && doc.for === stripLoras(prompt)) {
      setMarks(documentMarks(prompt, doc.elements))
    }
  }, [doc, prompt])

  useEffect(() => {
    const prose = stripLoras(prompt)
    if (!prose) {
      setMarks(NONE)
      useStore.getState().setDoc(null)
      return
    }
    // Already read, so do not read it again. **A re-ask cannot tell its own
    // words from the person's** — it is handed one string, and the second parse
    // of a sentence the model half wrote comes back claiming all of it as
    // derived, which quietly erases every grey run the first one earned. The
    // write below changes the prompt, so without this the feature undoes itself
    // one debounce after it works.
    //
    // It is also what makes a document survive a round trip through the box:
    // the state after a write is a document that already describes exactly what
    // is on screen, and there is nothing left to ask.
    if (useStore.getState().doc?.for === prose) return

    let live = true
    const t = setTimeout(async () => {
      const asked = prompt
      const r = await parse({ prose })
      if (!live || asked !== seen.current) return
      // A parse that failed, or that was refused as untrustworthy and came back
      // empty, clears the document rather than leaving the last one standing.
      // A document is valid only for the prose it was derived from, and the
      // prose has moved — keeping it would be the stale state this is built to
      // make unrepresentable, arriving through the front door.
      if (failed(r) || !r.ok || !r.elements.length) {
        setMarks(NONE)
        useStore.getState().setDoc(null)
        return
      }
      const text = r.text ?? prose
      const store = useStore.getState()

      // The box gets the document's prose, which is the only way a mark can be
      // *inline*: what came back is not what was typed the moment the model
      // filled anything in, and marks placed on the old string would underline
      // whatever happened to be at those offsets.
      //
      // **`insertionOnly` no longer gates this, and the reason is the whole
      // feature.** It asked "did the model revise their words rather than add
      // to them" and refused the write when it had. That is the right question
      // for enhancement and has no answer under replacement, where revising is
      // the job — measured on real output it passed 5 of 19 documents and
      // refused 14, so whether your sentence was replaced on screen depended on
      // whether the model happened to preserve enough word order. Worse, the
      // refused 14 still *ran*: the box showed your words and the encoder got
      // something else, which is the one thing this must never do.
      //
      // A replacement has to be editable, and nothing is editable that is not
      // on screen. So it is always written, the person's sentence is one ⌘Z
      // away through `docUndo`, and what they wrote is kept in `doc.from` and
      // recorded in the sidecar as `prompt_original`.
      //
      // One thing can still stop the write, and it does not stop the document.
      if (composing.current) {
        // A write refused is not a document refused. It still describes this
        // prose, so it still reaches the run — only the box is left alone, and
        // the marks go where they can be found in what is actually on screen.
        //
        // `composing`: an IME has a candidate window open. The textarea is
        // controlled, so a write is a React value change and a value change
        // mid-composition destroys the composition buffer and the window with
        // it. The staleness guard cannot cover this and it is not the rare
        // case: an open candidate window is precisely a prose state that has
        // been *stable* for longer than the debounce while the user is
        // mid-word, so the parse timer is tuned to almost exactly the dwell
        // time of the thing it must not interrupt. Dropped rather than queued —
        // composition ends by changing the prose, so this would be stale at the
        // moment it became usable, and the next pause re-parses.
        store.setDoc({ for: prose, from: prose, elements: r.elements, text })
        return
      }


      // The tokens ride along. They are notation the interpreter never saw, so
      // they are not in `text` — and a write that silently deleted somebody's
      // LoRA stack would be the one thing worse than not writing at all. Put
      // back at the end, because a token's position in the main prompt means
      // nothing to the backend, which reads them into a stack.
      // Was `[text, ...loraSyntax(asked)]`; see `Reroll`. Chips are not text.
      const next = text.trim()
      const caret = el.current?.selectionStart ?? null
      if (caret != null) caretTo.current = remapCaret(caret, asked, next)
      // **Keyed to what the box now holds**, not to what was typed a moment
      // ago. The write replaced the prose, so `for` has to move with it or the
      // document is stale against the very text it produced — `docFor` would
      // return null on the next Generate and the run would go plain with a
      // perfectly good document sitting in the store.
      store.applyDoc(next, { for: stripLoras(next), from: prose,
                             elements: r.elements, text })
      seen.current = next
    }, PAUSE_MS)
    return () => { live = false; clearTimeout(t) }
  }, [prompt])

  return marks
}

export type Run = { text: string; invented: boolean; span: boolean }

/**
 * The prompt cut where either channel changes, so each piece has one treatment.
 *
 * Cut on the union of both boundary sets rather than on the marks alone: a grey
 * run is nested *inside* an element's underline, so a slicer that only knew
 * about one of them would have to choose which of the two the overlap belongs
 * to, and either choice is wrong somewhere on the line.
 */
export function runs(text: string, marks: Marks): Run[] {
  const cuts = new Set<number>([0, text.length])
  for (const [a, b] of [...marks.invented, ...marks.spans]) {
    cuts.add(Math.max(0, Math.min(text.length, a)))
    cuts.add(Math.max(0, Math.min(text.length, b)))
  }
  const at = [...cuts].sort((x, y) => x - y)
  const inside = (list: Mark[], a: number, b: number) =>
    list.some(([x, y]) => x <= a && b <= y)

  const out: Run[] = []
  for (let i = 0; i < at.length - 1; i++) {
    const [a, b] = [at[i]!, at[i + 1]!]
    out.push({ text: text.slice(a, b),
               invented: inside(marks.invented, a, b),
               span: inside(marks.spans, a, b) })
  }
  // The trailing run is always emitted, even empty: a mirror that ends exactly
  // at its last mark loses the newline a user just typed, and the copy behind
  // the box stops matching the box by one line.
  if (!out.length || out[out.length - 1]!.text !== '') {
    out.push({ text: '', invented: false, span: false })
  }
  return out
}
