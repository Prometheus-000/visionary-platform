import { useState } from 'react'

import { Storyline } from './Storyline'
import { segment } from './segment'
import { mod, type Module } from './model'
import './storyline.css'

/**
 * A standalone page for the storyline, so the interaction can be held and
 * argued with before it displaces the prompt field in the real console.
 *
 * The presets are the cases the model was derived from and the ones that broke
 * it: the corridor, the detached third subject, and light with nowhere to hang.
 */
const PRESETS: Record<string, () => Module[]> = {
  'corridor (three figures)': () => {
    const woman = mod('On the left stands a tall woman')
    return [
    mod('Three figures standing in a hotel corridor'),
    mod('pale blue floral wallpaper, honey-coloured wooden door frames and a deep blue carpet running away from the camera', 'invented'),
    woman,
    mod('Beside her, to her right, stand two small girls of about eight', 'derived', [woman.id]),
    mod('All three face the camera directly, expressionless', 'invented'),
    mod('Even shadowless light with no visible source'),
  ] },
  'the detached third subject': () => [
    mod('Three people in a bar at night'),
    mod('Person A is sitting with person B, she is happy'),
    mod('Person B is ambivalent'),
    mod('Person C is at the far end of the room'),
  ],
  'light with nowhere to hang': () => [
    mod('A woman at a kitchen table'),
    mod('Hard side light'),
  ],
  'empty': () => [mod('')],
}

export function Sandbox() {
  const [mods, setMods] = useState<Module[]>(() => PRESETS['corridor (three figures)']!())
  const [raw, setRaw] = useState('')

  return (
    <div className="sb">
      <header className="sb-head">
        <b>Storyline</b>
        <span className="sb-sub">what your words will actually do</span>
        <span className="sb-presets">
          {Object.keys(PRESETS).map((k) => (
            <button key={k} type="button" onClick={() => setMods(PRESETS[k]!())}>
              {k}
            </button>
          ))}
        </span>
      </header>

      <section className="sb-panel">
        <Storyline mods={mods} setMods={setMods} />
      </section>

      <section className="sb-panel sb-paste">
        <label htmlFor="sb-raw">Or paste a prompt</label>
        <textarea
          id="sb-raw" rows={3} value={raw}
          placeholder="Three young people crowded into a small 1970s public bathroom…"
          onChange={(e) => setRaw(e.target.value)}
        />
        <button type="button" onClick={() => raw.trim() && setMods(segment(raw))}>
          Segment
        </button>
        <p>
          A stand-in for <code>/api/parse</code>. It breaks on sentence ends and nothing
          else — deciding what is an anchor and what hangs off it is the real parse, and
          that instruction is still blocked on experiments.
        </p>
      </section>
    </div>
  )
}
