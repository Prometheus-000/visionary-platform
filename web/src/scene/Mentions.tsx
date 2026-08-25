import { useEffect, useLayoutEffect, useRef, useState } from 'react'
import { createPortal } from 'react-dom'

import { useStore } from '../store'
import { handleOf } from './model'
import { hydrate, listCharacters, type SavedCharacter } from './arsenal'

/**
 * The picker you get by typing `@`, and the only way most people will ever make
 * a cast member.
 *
 * **This is `+ LoRA` inverted.** There, you open a picker in order to insert a
 * name. Here the name you are already typing *is* the picker — so the cast has no
 * empty state, because it is never on screen before it exists. The gesture creates
 * it, and the menu costs the console nothing because it floats.
 *
 * It opens **above** the caret rather than below it, which is not a preference:
 * on glass the keyboard occupies the bottom half of the screen and the console is
 * already pinned to the bottom of the stage. Portalled and fixed, because the
 * console is `overflow:auto` and would otherwise clip it against its own top edge
 * — the rule that stops a full console pushing the canvas out of frame.
 */
export type Mention = {
  /** Index of the `@`. */
  at: number
  /** What has been typed after it, which may be empty — a bare `@` opens the menu. */
  query: string
  /** Where the caret is, so a rewrite can put it back after the handle. */
  caret: number
}

/**
 * The mention the caret is inside, or null.
 *
 * Scanned backwards from the caret rather than matched with the painting regex,
 * because that one needs at least one character after the `@` and a bare `@` is
 * exactly the moment this has to open. It stops at whitespace and at a second `@`,
 * so a caret in the middle of ordinary prose finds nothing.
 */
export function mentionAt(line: string, caret: number): Mention | null {
  for (let i = caret - 1; i >= 0; i--) {
    const c = line[i]!
    if (c === '@') {
      const query = line.slice(i + 1, caret)
      // Only handle characters. `@ava, both hands` with the caret after the comma
      // is not a mention being typed, it is a finished one with prose after it.
      return /^[a-z0-9_]*$/i.test(query) ? { at: i, query, caret } : null
    }
    if (/\s/.test(c)) return null
  }
  return null
}

/** The line with the mention replaced by a settled handle, and where the caret goes. */
export function complete(line: string, m: Mention, handle: string) {
  const head = `${line.slice(0, m.at)}@${handle}`
  return { value: head + line.slice(m.caret), caret: head.length }
}

export function MentionMenu({ anchorRef, mention, onPick, onClose }: {
  /** The caret marker in the row's mirror — an exact position, because the mirror
   *  is a glyph-for-glyph copy of the textarea. Nothing else on the page knows
   *  where a caret is inside a `<textarea>`.
   *
   *  **The ref rather than the element**, because the marker and this menu are
   *  committed in the same pass: read during the parent's render, `.current` is
   *  still null and nothing re-renders to correct it, so the menu places itself in
   *  the top-left corner of the window and stays there. Refs are attached before
   *  any layout effect runs, so reading it in one is the first honest moment. */
  anchorRef: React.RefObject<HTMLElement | null>
  mention: Mention
  onPick: (handle: string) => void
  onClose: () => void
}) {
  const s = useStore()
  const box = useRef<HTMLDivElement>(null)
  const [at, setAt] = useState<{ left: number; top: number } | null>(null)
  const [sel, setSel] = useState(0)
  // The Arsenal, fetched when the menu opens. Names and notes only, at popover
  // speed; a saved character's files come one at a time when somebody picks.
  const [saved, setSaved] = useState<SavedCharacter[]>([])
  useEffect(() => { void listCharacters().then(setSaved) }, [])

  const q = handleOf(mention.query)
  const hits = s.scene.cast
    .filter((c) => c.name && (!q || c.name.startsWith(q)))
    .map((c) => ({ label: `@${c.name}`, note: c.note || c.kind, run: () => { onPick(c.name) } }))
  // A name already taken is not offered as new. `addCast` would settle the
  // collision by appending a number, which is right for a second person genuinely
  // called Ava and absurd as the only row under an exact match.
  const taken = new Set(s.scene.cast.map((c) => c.name))
  // **One row, because there is one kind.** This offered three — New character,
  // New place, New thing — which was our own vocabulary asked as a question:
  // ref-en §2.1 has a single `<Subject N>` covering people, environments,
  // clothing, styles and poses alike. Being made to classify a name before
  // writing about it is the closed vocabulary in its purest form, and what the
  // thing *is* belongs in its description, which is where the guide puts it.
  const makes = q && taken.has(q) ? [] : [{
    label: q ? `New subject “${q}”` : 'New subject',
    note: '',
    run: () => {
      const member = s.addCast('subject', q)
      onPick(member.name)
      // Opened straight onto its card, because a member made this way has a name
      // and nothing else, and a photograph is what turns the name into a subject.
      s.setRailOpen(member.id)
    },
  }]
  // The Arsenal's rows — the recall half of "saved deliberately, recalled by
  // typing". A saved character already in this scene's cast is not offered
  // twice; picking one creates the member now (the caret needs its handle) and
  // hydrates the files behind it, onto a card already open to watch them land.
  const recalls = saved
    .filter((c) => c.handle && !taken.has(c.handle)
                   && (!q || c.handle.startsWith(q)))
    .map((c) => ({
      label: `@${c.handle}`,
      note: c.note ? `saved — ${c.note}` : 'saved',
      run: () => {
        const member = s.addCast('subject', c.handle)
        onPick(member.name)
        s.setRailOpen(member.id)
        void hydrate(member.id, c)
      },
    }))
  const items = [...hits, ...recalls, ...makes]

  useLayoutEffect(() => {
    const el = box.current
    const anchor = anchorRef.current
    if (!el || !anchor) return
    const r = anchor.getBoundingClientRect()
    const w = el.offsetWidth
    const h = el.offsetHeight
    const left = Math.max(8, Math.min(r.left, window.innerWidth - w - 8))
    const top = Math.max(8, r.top - h - 6)
    setAt((cur) => (cur && cur.left === left && cur.top === top ? cur : { left, top }))
  })

  // Reset the highlight whenever the list changes under it, or a keystroke that
  // narrows the matches leaves the selection pointing past the end.
  useEffect(() => { setSel(0) }, [mention.query, s.scene.cast.length])

  useEffect(() => {
    const key = (e: KeyboardEvent) => {
      if (e.key === 'Escape') { e.preventDefault(); onClose(); return }
      if (e.key === 'ArrowDown' || e.key === 'ArrowUp') {
        e.preventDefault()
        setSel((n) => (n + (e.key === 'ArrowDown' ? 1 : items.length - 1)) % items.length)
        return
      }
      // ⏎ and ⇥ both take the highlighted row. ⏎ is what the row's own handler
      // would otherwise read as "start the next shot", so this listener runs in
      // the capture phase to get there first — a menu open over the caret is the
      // more specific claim on the key.
      if (e.key === 'Enter' || e.key === 'Tab') {
        e.preventDefault()
        e.stopPropagation()
        items[sel]?.run()
      }
    }
    document.addEventListener('keydown', key, true)
    return () => { document.removeEventListener('keydown', key, true) }
  }, [items, sel, onClose])

  if (!items.length) return null
  return createPortal(
    <div ref={box} className="tment"
         style={{ left: at?.left ?? 0, top: at?.top ?? 0,
                  visibility: at ? 'visible' : 'hidden' }}>
      {items.map((it, i) => (
        <button key={it.label} type="button" className={i === sel ? 'on' : undefined}
                // mousedown, not click: the textarea has focus and a click would
                // blur it first, which closes this on the way to firing.
                onMouseDown={(e) => { e.preventDefault(); it.run() }}
                onMouseEnter={() => { setSel(i) }}>
          {it.label}
          {it.note && <span className="hint">{it.note}</span>}
        </button>
      ))}
    </div>,
    document.body,
  )
}
