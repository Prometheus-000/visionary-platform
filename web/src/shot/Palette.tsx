import { useMemo } from 'react'

import { Popover } from '../ui/Popover'
import { shotItem, useStore } from '../store'
import { Glyph } from './Glyph'
import { shotLive, shotWhy } from './vocab'

/**
 * The closed vocabulary, behind one icon.
 *
 * H3 does not read a paragraph; it reads a document with named fields, published
 * in the model repo. The composer offered a textarea for it, so where camera
 * direction goes, whether tone belongs in the sentence, and what a comma does were
 * all things you found out by spending three minutes rendering. None of that is a
 * grammar anyone could infer — it is a schema — so the page emits it and the tiles
 * are how you say what goes in it.
 *
 * The vocabulary is served, never written into this file. A copy here would be a
 * second source of truth, and the first pill added on one side and not the other
 * would compile to "No such shot pill" against the page that offered it.
 */
export function Palette({ anchor, onClose }: { anchor: HTMLElement | null; onClose: () => void }) {
  const s = useStore()
  const groups = useMemo(() => {
    // Live groups first, dead ones after. Dimming rather than hiding is the
    // established call — a control that vanishes teaches only that the page lost
    // it, and "this app moves the camera on video" is worth learning while you are
    // on the image side. But on Krea 2 that is fifty-seven of eighty-seven tiles,
    // and leaving them in vocabulary order meant scrolling past two dead groups to
    // reach a live one. Display order only: the compiler still reads SHOT_VOCAB in
    // its own order, which is what decides clause position.
    return (s.state?.shot_vocab ?? []).slice()
      .sort((a, b) => (shotLive(s, a) ? 0 : 1) - (shotLive(s, b) ? 0 : 1))
  }, [s])

  return (
    <Popover anchor={anchor} className="pal" onClose={onClose}>
      {groups.map((g) => {
        const off = !shotLive(s, g)
        return (
          <section key={g.key} className={off ? 'off' : ''}>
            <h4>
              {g.label}
              {off && <i>{shotWhy(s)}</i>}
            </h4>
            <div className="tiles">
              {g.items.map((it) => {
                const k = `${g.key}.${it.key}`
                const dead = !shotLive(s, g, it)
                return (
                  <button key={k} type="button" disabled={dead}
                          className={`tl${s.shot.some((p) => p.key === k) ? ' on' : ''}${dead ? ' off' : ''}`}
                          // The tooltip is the phrase the pill writes, so hovering
                          // a tile shows the exact wording it puts in the
                          // document. That is the one thing a glyph and a two-word
                          // label between them still cannot say, and it is what
                          // makes the palette teach the phrasing rather than
                          // replace it.
                          title={it.phrase || it.hint || ''}
                          onClick={() => {
                            const added = !s.shot.some((p) => p.key === k)
                            s.toggleShot(k)
                            // A valued pill arrives with somewhere to type, and
                            // that is on the rail — so adding one closes the
                            // palette rather than leaving a caret behind a
                            // popover. Everything else leaves it open, because you
                            // are picking several and a palette that shuts on the
                            // first click is one you reopen five times.
                            if (added && shotItem(s.state?.shot_vocab ?? [], k)?.valued) onClose()
                          }}>
                    <Glyph cls={it.glyph} />
                    <span>{it.label}</span>
                  </button>
                )
              })}
            </div>
          </section>
        )
      })}
    </Popover>
  )
}
