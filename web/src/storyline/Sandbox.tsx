import { useState } from 'react'

import { Storyline } from './Storyline'
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

      {/* No separate "paste a prompt" control. `setText` splits on line
          breaks, so pasting a scene straight into the storyline segments it —
          a second box for the same job was a second door to one room. */}
      <section className="sb-panel">
        <Storyline mods={mods} setMods={setMods} />
      </section>

    </div>
  )
}
