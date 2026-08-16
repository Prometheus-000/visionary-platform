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
 * Train: a board of sessions, and the sets they train on.
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
 */
export function Train({ sess, onLightbox }: {
  sess: ReturnType<typeof useSessions>
  onLightbox: (src: string) => void
}) {
  const state = useStore((s) => s.state)
  const ds = useDatasets()
  // Two screens on one stage: the board, and the set you clicked through to.
  // Explicit rather than keyed on `ds.open` being null, because "+ New set"
  // lands on the sets screen with nothing open — which is the drop target, and
  // is the whole point of that path.
  const [screen, setScreen] = useState<'board' | 'sets'>('board')
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
    <div className="view studio" id="v-train">
      <div className="stage">
        <div className="canvas" id="t-canvas">
          {screen === 'sets' ? (
            /* Navigation, top-left of the thing it leaves — and inside the
               editor's own sticky bar, because a set is a long scroll and a way
               back that scrolls away is one you have to scroll up to find. The
               set is a place you clicked into from a card, so getting back is
               one word rather than a mode to un-toggle. */
            <Editor ds={ds} onLightbox={onLightbox}
                    lead={
                      <button className="s" id="ds-back" type="button"
                              onClick={() => setScreen('board')}>
                        ‹ Sessions
                      </button>
                    } />
          ) : (
            <div id="sess-board">
              <div className="opts board-head" style={{ marginTop: 0 }}>
                <b style={{ fontSize: 14 }}>Training</b>
                <span className="muted">
                  {active ? `${active} running`
                    : `${sess.rows.length} session${sess.rows.length === 1 ? '' : 's'}`}
                  {/* The cap, said out loud when it bites. A bounded listing
                      that does not mention the bound is a board that looks
                      like the whole board and is not. */}
                  {sess.total > sess.rows.length ? ` · showing ${sess.rows.length} of ${sess.total}` : ''}
                </span>
                <span className="actions">
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
                               onStart={() => void sess.start(s.id)}
                               onStop={() => void sess.stop(s.id)}
                               onDelete={() => void sess.remove(s.id)}
                               onOpenDataset={() => openSet(s.dataset)} />
                ))}
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Every set you already have, always open — the second way into the one
          screen that edits them. */}
      <aside className="drawer" id="ds-drawer">
        <div className="drawer-in">
          <div className="drawer-head">
            <span className="grow" />
            <button className="s" id="ds-fresh" type="button"
                    onClick={() => { setScreen('sets'); void ds.choose(null) }}>
              + New set
            </button>
          </div>
          {ds.error && <div className="err-box">{ds.error}</div>}
          <div id="ds-list" className="grid">
            <Rail ds={ds} onOpen={openSet} />
          </div>
        </div>
      </aside>

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
