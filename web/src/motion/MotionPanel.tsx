import { useEffect } from 'react'

import { failed } from '../api/client'
import { motion } from '../api/routes'
import { stripLoras } from '../lora/tokens'
import { Glyph } from '../shot/Glyph'
import { shotLive, shotWhy } from '../shot/vocab'
import { Popover } from '../ui/Popover'
import { motionLive, useStore } from '../store'

/** The server's own cap on a valued pill (`SHOT_VALUE_MAX`) — mirrored so a
 *  long run of sound picks degrades here, visibly, rather than being clamped
 *  on the far side of the network. */
const PILL_VALUE_MAX = 400

/**
 * Grounded motion suggestions, grouped, each one gesture from the prompt.
 *
 * **Three sections write prose, two write pills, one is vocabulary — and the
 * split is the H3 grammar, not taste.** Subjects, Environment and Camera are
 * clauses of the visual description, so picking one composes it into the
 * prompt box through `toggleMotion` — visible, editable, ⌘Z-undoable.
 * Sound and Dialogue are H3's *own named fields* (`overall_soundscape`, the
 * `<d>…</d>` dialogue line), so a pick lands as the valued pills that already
 * compile there — `sound.other` and `say.dialogue` — which keeps the audio out
 * of Wan's prompt (`needs: "audio"`) and on the rail where it is editable.
 * Score keeps the served tiles: a soundtrack is non-diegetic — the one thing
 * the frame cannot ground — and "no score" is worth the whole feature.
 *
 * **Fetch on open, cached at temperature 0.** The door press is the button the
 * user asked for; re-pressing is not a reroll (the decode is greedy), so the
 * answer is kept until the frame, the model or the grounding prose changes.
 * While picks are active the cache key holds the *base* prose — the composed
 * picks are downstream of the suggestions, and keying on them would refetch
 * on every toggle.
 */
export function MotionPanel({ anchor, onClose }: {
  anchor: HTMLElement | null
  onClose: () => void
}) {
  const s = useStore()

  useEffect(() => {
    // `getState()` rather than the render closure: this runs once per open,
    // and the open is the moment that decides what gets grounded.
    const st = useStore.getState()
    if (st.motion.busy) return
    const grounding = motionLive(st)
      ? stripLoras(st.motion.base as string).trim()
      : stripLoras(st.prompt).trim()
    const frame = st.keyframe.first
    if (!grounding && !frame) return
    const key = `${st.vid.model}|${frame?.length ?? 0}|${grounding}`
    if (st.motion.sug && st.motion.for === key) return
    st.setMotion({ busy: true, error: null })
    void motion({ prose: grounding, model: st.vid.model, first_frame: frame })
      .then((r) => {
        const done = useStore.getState()
        if (failed(r)) return done.setMotion({ busy: false, error: r.error })
        if (!r.ok) {
          return done.setMotion({
            busy: false, error: r.error ?? 'The model did not answer.' })
        }
        done.setMotion({ busy: false, error: null, sug: r.groups, for: key })
      })
  }, [])

  const groups = s.state?.motion_groups ?? []
  const sug = s.motion.sug ?? {}
  const nothing = !stripLoras(s.prompt).trim() && !s.keyframe.first
  const empty = groups.every((g) => !sug[g.key]?.length)

  /* Sound picks accumulate into one `sound.other` pill, comma-joined — the
     compiled soundscape sentence comma-joins its parts anyway, so the pill
     reads the way the document will. Unpicking removes just that phrase, and
     an emptied pill leaves the rail. */
  const soundPill = () => useStore.getState().shot.find((p) => p.key === 'sound.other')
  const soundOn = (t: string) =>
    (s.shot.find((p) => p.key === 'sound.other')?.value ?? '').includes(t)
  const toggleSound = (t: string) => {
    const pill = soundPill()
    if (!pill) {
      s.toggleShot('sound.other')
      s.setPill('sound.other', { value: t })
      return
    }
    const parts = (pill.value ?? '').split(', ').filter(Boolean)
    const next = parts.includes(t) ? parts.filter((x) => x !== t) : [...parts, t]
    if (!next.length) return s.toggleShot('sound.other')
    s.setPill('sound.other', { value: next.join(', ').slice(0, PILL_VALUE_MAX) })
  }

  /* One line per document is the dialogue pill's own shape, so a pick replaces
     rather than stacks, and picking the active line takes it off. */
  const diaOn = (t: string) =>
    s.shot.find((p) => p.key === 'say.dialogue')?.value === t
  const toggleDialogue = (t: string) => {
    const pill = useStore.getState().shot.find((p) => p.key === 'say.dialogue')
    if (pill?.value === t) return s.toggleShot('say.dialogue')
    if (!pill) s.toggleShot('say.dialogue')
    s.setPill('say.dialogue', { value: t })
  }

  // Stale picks show unpicked — see `motionLive` for why a lying highlight is
  // worse here than it looks.
  const live = motionLive(s)
  const row = (g: { key: string }, t: string, i: number) => {
    const id = `${g.key}:${i}`
    const on = g.key === 'sound' ? soundOn(t)
      : g.key === 'dialogue' ? diaOn(t)
      : live && s.motion.picks.includes(id)
    const act = g.key === 'sound' ? () => toggleSound(t)
      : g.key === 'dialogue' ? () => toggleDialogue(t)
      : () => s.toggleMotion(id)
    return (
      <button key={id} type="button" className={`sug${on ? ' on' : ''}`}
              onClick={act}>
        {t}
      </button>
    )
  }

  const score = s.state?.shot_vocab.find((g) => g.key === 'score')
  const scoreOff = !score || !shotLive(s, score)

  return (
    <Popover anchor={anchor} className="pal motion" onClose={onClose}>
      {s.motion.busy && (
        <p className="muted mo-note">
          {s.keyframe.first ? 'Reading the frame…' : 'Reading the scene…'}
        </p>
      )}
      {!s.motion.busy && s.motion.error && (
        <p className="muted warn mo-note">{s.motion.error}</p>
      )}
      {!s.motion.busy && !s.motion.error && nothing && (
        <p className="muted mo-note">
          Type a scene or attach a first frame, then open this again.
        </p>
      )}
      {!s.motion.busy && !s.motion.error && !nothing && empty && (
        <p className="muted mo-note">
          No suggestions this time — describe the motion yourself.
        </p>
      )}
      {groups.map((g) => {
        const items = sug[g.key] ?? []
        if (!items.length) return null
        return (
          <section key={g.key}>
            <h4>{g.label}</h4>
            {items.map((t, i) => row(g, t, i))}
          </section>
        )
      })}
      {score && (
        <section className={scoreOff ? 'off' : ''}>
          <h4>
            {score.label}
            {scoreOff && <i>{shotWhy(s)}</i>}
          </h4>
          <div className="tiles">
            {score.items.map((it) => {
              const k = `${score.key}.${it.key}`
              return (
                <button key={k} type="button" disabled={scoreOff}
                        className={`tl${s.shot.some((p) => p.key === k) ? ' on' : ''}${scoreOff ? ' off' : ''}`}
                        title={it.phrase || it.hint || ''}
                        onClick={() => {
                          // The palette's own rule: a valued tile arrives with
                          // somewhere to type, and that is on the rail — so
                          // adding one closes the panel rather than leaving a
                          // caret behind a popover.
                          const added = !s.shot.some((p) => p.key === k)
                          s.toggleShot(k)
                          if (added && it.valued) onClose()
                        }}>
                  <Glyph cls={it.glyph} />
                  <span>{it.label}</span>
                </button>
              )
            })}
          </div>
        </section>
      )}
    </Popover>
  )
}
