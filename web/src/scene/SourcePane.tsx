import { useEffect, useLayoutEffect, useRef } from 'react'

import { useCompiled } from '../shot/useCompiled'
import { useStore } from '../store'

/** `#doc-source`'s line height and the shortest pane worth drawing a header on. */
const LINE = 21
const MIN_LINES = 3

/**
 * The compiled document, as view source.
 *
 * **The precedent is not a disclosure, it is devtools.** Reading the prompt was
 * frame 6 of the console ladder and that was wrong: it made reading what you had
 * compiled a *step in composing*. It is not one. The scene is the source and the
 * document is what it compiles to, so this is the same relationship a page has
 * to its markup — and four things follow from that rather than being chosen:
 *
 * - **No button says "inspect".** Devtools is a chord and a context menu and the
 *   page never advertises it, so the console carries nothing for this and costs
 *   not one pixel more than it did. ⌘⌥U, or right-click the console.
 * - **It is not in the console budget, because it is not in the console.** It
 *   takes the canvas the way devtools takes the window, and that is fine —
 *   nobody is judging a render while reading a prompt. The surface that kept
 *   breaking the budget stops competing for it.
 *
 *   **It takes what it needs of the canvas, not all of it.** `inset:0` was the
 *   first version and a six-line document then sat in a 680px box with five
 *   hundred pixels of black under it. Devtools docks to a *fraction* of the
 *   window and drags to more; this grows from the bottom edge — the side the
 *   console it describes is on — to fit what it holds, and scrolls once it
 *   reaches the top.
 * - **A textarea, not a `<pre>`.** Devtools is directly editable and the page
 *   responds live; this is too, and the edit is what runs. Where the precedent
 *   stops applying: a devtools edit evaporates on reload because the source file
 *   is the truth you are expected to port back to, and here that would mean a
 *   render you cannot reproduce. So editing **detaches** — one bit for the whole
 *   document, visible in the header, one gesture back. Not per-field pinning:
 *   that is six independent states nobody asked for, and an attempt to make a
 *   derived surface partly authoritative. Either the scene is driving it or you
 *   have taken it over, and you can always see which.
 * - **It shows derivation.** Devtools names the rule behind a computed value;
 *   put the caret in a `[Shot N]` block and that row lights up below.
 *
 * Reattaching recompiles and discards the edit, so the way back is exact — the
 * thing you typed is gone and what you had before is what you had before.
 */
export function SourcePane() {
  const s = useStore()
  const area = useRef<HTMLTextAreaElement>(null)
  // Only while attached. A detached document is yours, and refetching under it
  // would be the compiler quietly arguing with an edit it has been told to stay
  // out of — the request is also pointless work four times a second.
  const compiled = useCompiled(s.docOpen && s.doc === null)
  const text = s.doc ?? compiled

  useEffect(() => {
    const key = (e: KeyboardEvent) => {
      if (e.key === 'Escape') s.setDocOpen(false)
    }
    document.addEventListener('keydown', key)
    return () => { document.removeEventListener('keydown', key) }
  }, [s])

  // Grow to the document, up to the canvas; past that, scroll. The same shape as
  // `autoGrow`, and the `height = 'auto'` first is load-bearing for the same
  // reason — `scrollHeight` on an element with an explicit height reports that
  // height, so without it the pane can only ever get taller.
  useLayoutEffect(() => {
    const el = area.current
    const box = el?.parentElement
    const stage = box?.parentElement
    if (!el || !box || !stage) return
    el.style.height = 'auto'
    const chrome = box.getBoundingClientRect().height - el.getBoundingClientRect().height
    const cap = stage.getBoundingClientRect().height - chrome
    // A floor, so a one-field document is not a sliver with a header on it. Three
    // lines is the shortest thing that reads as a document rather than a caption.
    el.style.height = `${String(Math.max(MIN_LINES * LINE, Math.min(el.scrollHeight, cap)))}px`
  })

  /**
   * Which shot the caret is in, if any.
   *
   * The last `[Shot N]` marker at or before it — the same scan the document's own
   * reader does. Only meaningful while attached: a detached document is arbitrary
   * text and its markers are not claims about any row.
   */
  const derive = () => {
    if (s.doc !== null) return
    const at = area.current?.selectionStart ?? 0
    let n = -1
    for (const m of text.slice(0, at).matchAll(/\[Shot (\d+)\]/g)) n = Number(m[1]) - 1
    const row = s.scene.shots[n]
    if (row) s.selectShot(row.id)
  }

  return (
    <div className="tsource">
      <div className="tsource-h">
        {/* A sentence, not a status row. "COMPILED · REF2VA" was the devtools
            costume the owner never asked for — what this header owes the
            reader is whose words are running, said the way the app talks. */}
        <span>{s.doc === null ? 'What the model reads' : 'What the model reads — your edit'}</span>
        {s.doc !== null && (
          <button type="button" className="tag warn" onClick={() => { s.setDoc(null) }}>
            let the scene write it again
          </button>
        )}
        <button type="button" className="x" title="Close (Escape)"
                onClick={() => { s.setDocOpen(false) }}>×</button>
      </div>
      <textarea id="doc-source" ref={area} spellCheck={false} value={text}
                onChange={(e) => { s.setDoc(e.target.value) }}
                onKeyUp={derive} onClick={derive} onSelect={derive} />
    </div>
  )
}
