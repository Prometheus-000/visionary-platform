import { useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react'

import { field, heat } from './heat'
import {
  compile, dependsOnPrior, indent, insertAfter, move, outdent, remove, rows,
  setText, shares, tie, untie, type Module,
} from './model'

/**
 * The prompt, seen as the structure it already is.
 *
 * **The constraint that shaped this: it must not be harder than writing a
 * prompt.** A surface where you assemble elements to get a sentence has failed
 * before it renders, so this is a text field first — type into it and you get a
 * flat storyline, which is exactly a prompt, and every structural gesture is
 * optional.
 *
 * The structural gesture is **Tab**, borrowed from every bulleted list anyone
 * has used: indent an element and it becomes a dependent of the one above,
 * outdent and it leaves. That is merge and split, which is the whole relational
 * model — an element either hangs off an anchor or becomes one, and there is no
 * third state here either.
 *
 * What merging *does* is change the emitted grammar from two clauses to one.
 * Light indented under a subject reads "…, hard light raking across her face"
 * and is a property of her; the same words at the top level are their own
 * sentence and the light is a character. The heat shows it happen: shares are
 * summed over a subtree, so a merged element's extent folds into its anchor and
 * two warm spots become one, while splitting takes the warmth back out. Nothing
 * animates that — it falls out of the arithmetic.
 */
export function Storyline({
  mods, setMods, onSubmit,
}: {
  mods: Module[]
  setMods: (m: Module[]) => void
  onSubmit?: () => void
}) {
  const [focus, setFocus] = useState<string | null>(null)
  const refs = useRef<Record<string, HTMLTextAreaElement | null>>({})
  const railRef = useRef<HTMLDivElement | null>(null)
  const [centres, setCentres] = useState<Record<string, number>>({})

  const list = useMemo(() => rows(mods), [mods])
  const [tying, setTying] = useState<string | null>(null)
  /**
   * What a relation points at, named the way a person would name it.
   *
   * "On the left stands a tall woman" is referred to as *the tall woman*, not
   * as its first four words — so the leading placement clause comes off before
   * anything is shown. This is a stand-in with an obvious ceiling: naming what
   * an element is *about* is understanding it, which is the parse's job. The
   * heuristic covers the constructions the examples actually use and will look
   * silly outside them, which is the right kind of wrong for a placeholder.
   */
  const label = (m: Module) => {
    const stripped = m.text.trim()
      .replace(/^(on|in|at|to|beside|behind|near|opposite)\s+(the\s+)?\w+(\s+\w+)?,?\s*/i, '')
      .replace(/^(stands?|sits?|leans?|crouch(es)?|kneels?)\s+/i, '')
      .replace(/^(and\s+)?/i, '')
    const w = (stripped || m.text).trim().split(/\s+/).filter(Boolean)
    return w.length <= 4 ? w.join(' ') : `${w.slice(0, 4).join(' ')}…`
  }
  const byId = useMemo(
    () => Object.fromEntries(rows(mods).map((r) => [r.m.id, r.m])) as Record<string, Module>,
    [mods],
  )
  const top = useMemo(() => shares(mods), [mods])
  const maxShare = Math.max(...top, 0.0001)

  // Top-level elements own a share; a dependent's extent belongs to its anchor,
  // so it inherits the anchor's colour rather than getting one of its own.
  const shareOf = useMemo(() => {
    const out: Record<string, number> = {}
    list.forEach(({ m, path }) => { out[m.id] = top[path[0] ?? 0] ?? 0 })
    return out
  }, [list, top])

  useLayoutEffect(() => {
    const rail = railRef.current
    if (!rail) return
    const box = rail.getBoundingClientRect()
    if (!box.width) return
    const next: Record<string, number> = {}
    mods.forEach((m) => {
      const el = refs.current[m.id]
      if (!el) return
      const r = el.getBoundingClientRect()
      next[m.id] = ((r.left + r.width / 2 - box.left) / box.width) * 100
    })
    if (JSON.stringify(next) !== JSON.stringify(centres)) setCentres(next)
  }, [mods, list, centres])

  useEffect(() => {
    if (focus) refs.current[focus]?.focus()
  }, [focus, mods])

  useLayoutEffect(() => {
    for (const el of Object.values(refs.current)) {
      if (!el) continue
      el.style.height = 'auto'
      el.style.height = `${el.scrollHeight}px`
    }
  }, [list])

  const gradient = field(
    mods.map((m, i) => ({
      pct: centres[m.id] ?? ((i + 0.5) / Math.max(mods.length, 1)) * 100,
      color: heat(top[i] ?? 0, maxShare),
    })),
  )

  function key(e: React.KeyboardEvent, path: number[], m: Module) {
    const mk = (next: Module[]) => { setMods(next); setFocus(m.id) }
    if (e.key === 'Tab') {
      e.preventDefault()
      mk(e.shiftKey ? outdent(mods, path) : indent(mods, path))
    } else if (e.key === 'Enter' && !e.shiftKey && !e.metaKey) {
      e.preventDefault()
      const [next, id] = insertAfter(mods, path)
      setMods(next); setFocus(id)
    } else if (e.key === 'Enter' && e.metaKey) {
      e.preventDefault(); onSubmit?.()
    } else if (e.key === 'Backspace' && !m.text && list.length > 1) {
      e.preventDefault()
      const i = list.findIndex((r) => r.m.id === m.id)
      setMods(remove(mods, path)); setFocus(list[Math.max(0, i - 1)]?.m.id ?? null)
    } else if ((e.key === 'ArrowUp' || e.key === 'ArrowDown') && e.altKey) {
      e.preventDefault()
      mk(move(mods, path, e.key === 'ArrowUp' ? -1 : 1))
    }
  }

  return (
    <div className="sl">
      {/* The field, sampled at element centres. Continuous because the encoder
          is: one composition, not N independent weights. */}
      <div className="sl-heat" ref={railRef} style={{ background: gradient }} />

      <div className="sl-rows">
        {list.map(({ m, depth, path }) => (
          <div
            key={m.id}
            className={`sl-row${depth ? ' is-atom' : ''}`
              + `${m.origin === 'invented' ? ' is-invented' : ''}`
              + `${dependsOnPrior(m) ? ' is-tied' : ''}`}
            style={{ paddingLeft: `${depth * 22}px` }}
          >
            {(dependsOnPrior(m) || m.ties.length > 0) && (
              <span
                className={`sl-tied${m.ties.length ? ' is-linked' : ''}`}
                title={m.ties.length
                  ? `Goes with “${m.ties.map((t) => label(byId[t] ?? m)).join('”, “')}”`
                  : 'This mentions something above it, so it has to come after'}
              >↰</span>
            )}
            <span
              className="sl-tick"
              style={{ background: depth ? 'transparent' : heat(shareOf[m.id] ?? 0, maxShare) }}
              title={depth ? 'part of the element above' : `${Math.round((shareOf[m.id] ?? 0) * 100)}% of the picture`}
            />
            <textarea
              ref={(el) => { refs.current[m.id] = el }}
              className="sl-text"
              rows={1}
              value={m.text}
              placeholder={list.length === 1 ? 'Describe the shot…' : ''}
              onChange={(e) => {
                e.target.style.height = 'auto'
                e.target.style.height = `${e.target.scrollHeight}px`
                setMods(setText(mods, path, e.target.value))
              }}
              onKeyDown={(e) => key(e, path, m)}
            />
            {!depth && (
              <span className="sl-tools">
                {m.ties.map((t) => (
                  <button
                    key={t} type="button" className="sl-chip"
                    title="Separate these"
                    onClick={() => setMods(untie(mods, m.id, t))}
                  >with {label(byId[t] ?? m)}</button>
                ))}
                <button
                  type="button" className="sl-tie-btn"
                  title="Say this goes with something else in the shot"
                  onClick={() => setTying(tying === m.id ? null : m.id)}
                >goes with…</button>
                <span className="sl-share">{Math.round((shareOf[m.id] ?? 0) * 100)}%</span>
              </span>
            )}
            {tying === m.id && (
              <div className="sl-picker">
                {mods.filter((o) => o.id !== m.id && !m.ties.includes(o.id)).map((o) => (
                  <button
                    key={o.id} type="button"
                    onClick={() => { setMods(tie(mods, m.id, o.id)); setTying(null) }}
                  >{label(o)}</button>
                ))}
              </div>
            )}
          </div>
        ))}
      </div>

      <details className="sl-peek">
        <summary>what Krea 2 reads</summary>
        <pre>{compile(mods) || '—'}</pre>
      </details>

      {/* Says what the gestures do in the shot, never what they do to the
          model. "Tab" is not "indent" or "merge" here — it is "this describes
          the thing above", which is the only reading a storyteller needs. */}
      <p className="sl-hint">
        <b>Tab</b> — this describes the one above · <b>⇧Tab</b> — on its own again ·
        <b> ⌥↑↓</b> — move it in the frame
      </p>
    </div>
  )
}
