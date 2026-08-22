import { useEffect, useState } from 'react'

import { failed } from '../api/client'
import { reroll } from '../api/routes'
import { stripLoras } from '../lora/tokens'
import { useStore } from '../store'
import { derivedText, documentMarks, inventedAt } from './marks'

/**
 * The one thing a grey run can do that typing cannot.
 *
 * **A span is an object for exactly one purpose**, and this is it. Editing an
 * invented run is already just typing into it — `remap` drops the mark the edit
 * landed on, the words turn dark and become yours with no gesture and no
 * commit — so an inline editable rooted in the run would be a second text
 * surface competing with the one underneath it, buying nothing. What is left
 * that plain text cannot give is *ask for a different one*, which is the only
 * terms invention was ever allowed on.
 *
 * Rooted at the run's own end and revealed only while the caret is inside it.
 * Nothing at rest: no card, no popover, no chip, and nothing drawn over a
 * render — the console's own rule, and the reason this is a mark on the
 * sentence rather than a panel about it.
 */
export function Reroll({ mirror, el }: {
  mirror: React.RefObject<HTMLDivElement | null>
  el: React.RefObject<HTMLTextAreaElement | null>
}) {
  const s = useStore()
  const [caret, setCaret] = useState<number | null>(null)
  const [busy, setBusy] = useState(false)
  const [at, setAt] = useState<{ x: number; y: number } | null>(null)

  // The caret is not a React value, so it is watched rather than derived. Only
  // while the field has focus: a run does not stay armed because the caret was
  // left in it on the way to pressing Generate.
  useEffect(() => {
    const read = () => {
      const t = el.current
      setCaret(t && document.activeElement === t ? t.selectionStart : null)
    }
    document.addEventListener('selectionchange', read)
    window.addEventListener('blur', read)
    read()
    return () => {
      document.removeEventListener('selectionchange', read)
      window.removeEventListener('blur', read)
    }
  }, [el])

  const doc = s.doc
  const hit = doc && caret != null && doc.for === stripLoras(s.prompt)
    ? inventedAt(s.prompt, doc.elements, caret) : null

  // Measured off the mirror, which is the only thing on screen that knows where
  // a character is: the textarea cannot report the position of an offset, and
  // the mirror is already a glyph-exact copy sitting at the same metrics.
  useEffect(() => {
    if (!hit || !mirror.current) { setAt(null); return }
    const r = rectAt(mirror.current, hit.run[1])
    const box = mirror.current.getBoundingClientRect()
    setAt(r ? { x: r.right - box.left, y: r.top - box.top } : null)
  }, [hit, mirror, s.prompt])

  if (!hit || !at || !doc) return null

  const press = async () => {
    setBusy(true)
    // **The user's words, not the box.** By now the box holds the document's
    // prose, model's clauses and all, and sending that would ask the validator
    // whether the model's own words are in the model's own words. See
    // `derivedText`.
    const box = stripLoras(s.prompt)
    const prose = derivedText(box, documentMarks(box, doc.elements).invented)
    const r = await reroll({ prose, document: doc.elements, only: hit.id })
    // **All three outcomes settle the same way**, and that is deliberate. A
    // reroll lands as new text, as the same text, or as a rejection; two of
    // those change nothing on screen, and a flicker that fired for one but not
    // the other would build a channel telling you which way the validator
    // went — the one thing the silent degrade exists not to say. What the
    // press reports is the press.
    setBusy(false)
    if (failed(r) || !r.ok || !r.elements.length) return

    const text = r.text ?? box
    // No `insertionOnly` gate here either, and for the reason the parse write
    // gives: a reroll is a request to write this differently, so refusing it
    // for having written it differently refuses the button. The old document is
    // one ⌘Z away and `from` still points at the sentence behind both.
    // Was `[text, ...loraSyntax(s.prompt)]` — the tokens rode along because a
    // write that replaced the box would otherwise delete them. A LoRA is a chip
    // now and lives nowhere near this string.
    const next = text.trim()
    useStore.getState().applyDoc(next,
      { for: stripLoras(next), from: useStore.getState().doc?.from ?? prose,
        elements: r.elements, text })
  }

  return (
    <button type="button" className="mk-reroll" disabled={busy}
            style={{ left: `${at.x}px`, top: `${at.y}px` }}
            title="Read this one again"
            // The caret must survive the press, or leaving the button dismisses
            // the thing it belongs to before the click lands.
            onMouseDown={(e) => e.preventDefault()}
            onClick={() => void press()}>
      <svg viewBox="0 0 16 16" width="11" height="11" aria-hidden="true">
        <path d="M13 8a5 5 0 1 1-1.5-3.6M13 2v3h-3" fill="none"
              stroke="currentColor" strokeWidth="1.6"
              strokeLinecap="round" strokeLinejoin="round" />
      </svg>
    </button>
  )
}

/** Where a character offset actually sits, as a rect in the mirror. */
function rectAt(root: HTMLElement, offset: number): DOMRect | null {
  const walk = document.createTreeWalker(root, NodeFilter.SHOW_TEXT)
  let seen = 0
  let node: Node | null
  while ((node = walk.nextNode())) {
    const len = node.textContent?.length ?? 0
    if (seen + len >= offset) {
      const range = document.createRange()
      range.setStart(node, Math.max(0, offset - seen))
      range.setEnd(node, Math.max(0, offset - seen))
      const rects = range.getClientRects()
      return rects.length ? rects[rects.length - 1]! : range.getBoundingClientRect()
    }
    seen += len
  }
  return null
}
