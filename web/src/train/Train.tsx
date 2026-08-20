import { useEffect, useState } from 'react'

import { thumbUrl } from '../api/routes'
import { Editor } from '../datasets/Editor'
import { useDatasets } from '../datasets/useDatasets'
import { useStore } from '../store'
import { SessionCard } from './SessionCard'
import { SessionForm } from './SessionForm'
import { type Draft, isActive, paramsToForm, type useSessions } from './useSessions'
import type { Session, TrainParams } from '../api/types'

/**
 * Train: a board of sessions, and — behind one door — the sets they train on.
 *
 * **A run is a card, and there is no page for one.** The console under the
 * contact sheet is gone — it held a single `job` in component state, which made
 * "one run at a time" a property of the page rather than of the backend, and
 * `train_job` never shared anything between runs. A board is what the backend
 * could always do; four cards is four containers and four bills, and nothing to
 * coordinate.
 *
 * **Making a set is untangled from starting a run.** They were one screen
 * because a run needed a set in front of it; now the set is a menu in the form,
 * with "+ New set" as its last option — so the set you are building and the run
 * you are setting up stop being the same act, and the half-finished card is
 * what carries you between them.
 *
 * **And the sets moved out of the board's margin.** They were a permanent 320px
 * rail beside the cards, which was the last thing left over from the screen
 * where a run needed a set in front of it. Nothing on the board reads them: the
 * form picks a set from a menu, and a card names its own and links to it. A rail
 * that no decision on the screen consults is 320px spent on a list you are not
 * using, so the sets got a screen of their own — index and drop target together,
 * which is what the rail and the empty editor each had half of.
 */
export function Train({ sess, onLightbox, screen, setScreen }: {
  sess: ReturnType<typeof useSessions>
  onLightbox: (src: string) => void
  /** Two screens on one stage: the board, and the sets. Explicit rather than
   *  keyed on `ds.open` being null, because the sets screen with nothing open
   *  is the index *and* the drop target, and is where "+ New set" lands. Owned
   *  by App rather than here, because the header's Sets door has to be able to
   *  land on the sets screen directly — a high-level feature is one the front
   *  door reaches, not one found two levels into Training. */
  screen: 'board' | 'sets'
  setScreen: (s: 'board' | 'sets') => void
}) {
  const state = useStore((s) => s.state)
  const ds = useDatasets()
  const [form, setForm] = useState<Draft | null>(null)
  const [saving, setSaving] = useState(false)

  useEffect(() => {
    void ds.load()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  const openSet = (name: string) => {
    setScreen('sets')
    void ds.choose(name)
  }

  const blank = (): Draft => ({
    lora_name: '', trigger_word: '', dataset: '',
    params: paramsToForm(undefined, (state?.train_defaults ?? {}) as TrainParams),
  })

  const edit = (s: Session): Draft => ({
    id: s.id, lora_name: s.lora_name, trigger_word: s.trigger_word, dataset: s.dataset,
    params: paramsToForm(s.params, (state?.train_defaults ?? {}) as TrainParams),
  })

  const save = async (d: Draft, start: boolean) => {
    setSaving(true)
    const saved = await sess.save(d)
    setSaving(false)
    if (!saved) return
    setForm(null)
    // The id off the reply, not out of `sess.rows`: the reload has happened but
    // this closure still holds the list from before it, so a freshly created
    // card is not in there to be found.
    if (start && saved.session?.id) await sess.start(saved.session.id)
  }

  const active = sess.rows.filter(isActive).length

  return (
    <div className="view studio nodrawer" id="v-train">
      <div className="stage">
        <div className="canvas" id="t-canvas">
          {screen === 'sets' ? (
            <div id="sets-screen">
              {/* The screen's own head, carrying the way out. With a set open
                  this is not rendered: the Editor puts the back button inside
                  its own sticky bar instead, because a set is a long scroll and
                  a way back that scrolls away is one you scroll up to find. */}
              {!ds.open && (
                <div className="opts board-head" style={{ marginTop: 0 }}>
                  <button className="t" id="ds-back" type="button"
                          onClick={() => setScreen('board')}>
                    ‹ Sessions
                  </button>
                  <h2>Sets</h2>
                  <span className="muted">
                    {ds.loading && !ds.rows.length
                      ? 'Loading…'
                      : `${ds.rows.length} set${ds.rows.length === 1 ? '' : 's'}`}
                  </span>
                </div>
              )}

              {/* No "+ New set" anywhere on this screen: the drop target is
                  directly below and making a set *is* dropping on it. A button
                  whose whole effect is to reveal the thing already on screen is
                  the kind this pass exists to remove. */}
              <Editor ds={ds} onLightbox={onLightbox}
                      lead={ds.open ? (
                        <button className="t" id="ds-back" type="button"
                                onClick={() => void ds.choose(null)}>
                          ‹ Sets
                        </button>
                      ) : undefined} />

              {/* The index, under the drop target and only when nothing is
                  open: with a set on screen this is the set you are looking at,
                  and a list of the others beneath it is a rail again. */}
              {!ds.open && (
                <div id="ds-index">
                  {ds.error && <div className="err-box">{ds.error}</div>}
                  {/* Said out loud, not left blank: the listing walks the
                      volume, and an empty screen for the length of that walk
                      reads as "no sets" — which is a lie about a library that
                      merely has not answered yet. Only on the first-ever
                      fetch; after that the cached rows paint instantly. */}
                  {ds.loading && !ds.rows.length ? (
                    <p className="muted" id="ds-loading" style={{ margin: '18px 2px' }}>
                      Loading your sets…
                    </p>
                  ) : (
                    <div id="ds-list" className="grid">
                      <Rail ds={ds} onOpen={openSet} />
                    </div>
                  )}
                </div>
              )}
            </div>
          ) : (
            <div id="sess-board">
              <div className="opts board-head" style={{ marginTop: 0 }}>
                <h2>Training</h2>
                <span className="muted">
                  {active ? `${active} running`
                    : `${sess.rows.length} session${sess.rows.length === 1 ? '' : 's'}`}
                  {/* The cap, said out loud when it bites. A bounded listing
                      that does not mention the bound is a board that looks
                      like the whole board and is not. */}
                  {sess.total > sess.rows.length ? ` · showing ${sess.rows.length} of ${sess.total}` : ''}
                </span>
                <span className="actions">
                  {/* A door, so it takes a word rather than a glyph — and a text
                      button, because there is exactly one action on this screen
                      worth a filled one and it is the next thing along. */}
                  <button className="t" id="ds-door" type="button"
                          onClick={() => { setScreen('sets'); void ds.choose(null) }}>
                    Sets
                  </button>
                  <button className="b" id="new-session" type="button"
                          onClick={() => setForm(blank())}>
                    + Create session
                  </button>
                </span>
              </div>

              {sess.error && <div className="err-box">{sess.error}</div>}

              {!sess.rows.length && sess.loaded && (
                <div className="blank" id="sess-empty">
                  <div>
                    <b>No sessions yet.</b>
                    <p className="muted" style={{ marginTop: 8 }}>
                      A session is a LoRA, a set and its dials. Start as many as you
                      like — each run gets its own GPU.
                    </p>
                  </div>
                </div>
              )}

              <div id="sess-list">
                {sess.rows.map((s) => (
                  <SessionCard key={s.id} s={s}
                               ds={ds.rows.find((r) => r.name === s.dataset)}
                               onEdit={() => setForm(edit(s))}
                               onStop={() => void sess.stop(s.id)}
                               onDelete={() => void sess.remove(s.id)}
                               onOpenDataset={() => openSet(s.dataset)} />
                ))}
              </div>
            </div>
          )}
        </div>
      </div>

      {form && (
        <SessionForm initial={form} state={state} datasets={ds.rows}
                     saving={saving} error={sess.error}
                     onSave={(d, start) => void save(d, start)}
                     onClose={() => setForm(null)}
                     onNewDataset={(d) => {
                       // Saved first, then out of the way: the card is what you
                       // come back to, and a form abandoned mid-way is a form
                       // you retype.
                       void sess.save(d).then(() => {
                         setForm(null)
                         setScreen('sets')
                         void ds.choose(null)
                       })
                     }} />
      )}
    </div>
  )
}

/** Drafts first: they are the set you are working on, and the reason they are labelled at
 *  all is that the label is a promise about what happens to them. */
function Rail({ ds, onOpen }: { ds: ReturnType<typeof useDatasets>; onOpen: (n: string) => void }) {
  const drafts = ds.rows.filter((d) => !d.saved)
  const saved = ds.rows.filter((d) => d.saved)

  const card = (d: (typeof ds.rows)[number]) => (
    <div className="ds-row" key={d.name}>
      <button className={['ds-card', d.name === ds.open ? 'sel' : '', d.saved ? '' : 'draft']
                .filter(Boolean).join(' ')}
              type="button" onClick={() => onOpen(d.name)}>
        {d.cover
          ? <img className="ds-cover" loading="lazy" src={thumbUrl(d.name, d.cover)} alt="" />
          : <div className="ds-cover empty">▤</div>}
        <div className="ds-meta">
          <b>{d.name}</b>
          <div className="muted" style={{ marginTop: 3, fontSize: 12 }}>
            {d.count} image{d.count === 1 ? '' : 's'}
            {/* Counted apart, because only one of the two can be trained on
                today — see the TODO at `train_job`. */}
            {d.videos ? ` · ${d.videos} clip${d.videos === 1 ? '' : 's'}` : ''}
            {d.uncaptioned ? ` · ${d.uncaptioned} uncaptioned` : ''}
          </div>
        </div>
      </button>
      <button className="ds-x" title="Delete" type="button"
              onClick={() => void ds.remove(d.name)}>×</button>
    </div>
  )

  return (
    <>
      {!!drafts.length && (
        <p className="ds-group">Unsaved <span>· cleared when you close the app</span></p>
      )}
      {drafts.map(card)}
      {!!saved.length && !!drafts.length && <p className="ds-group">Saved</p>}
      {saved.map(card)}
    </>
  )
}
