import { useState } from 'react'

import type { Session } from '../api/types'
import type { DatasetRow } from '../datasets/useDatasets'
import { Sheet } from '../ui/Sheet'
import { isActive } from './useSessions'

/**
 * One training run, as a card.
 *
 * The card is what replaced the full-page run console, and the reason is the
 * backend rather than the layout: `train_job` never shared anything between
 * runs, so "one at a time" was a property of a page that could only hold one
 * `job` in a variable. A board of cards is what the backend was always able to
 * do.
 *
 * **What a card says is what the terminal says.** A bar alone cannot tell
 * training from stuck, and the numbers that tell them apart are the ones tqdm
 * prints: step over total, epoch over total, the rate — `it/s` or `s/it`,
 * whichever way up it came, because which one it is is information — elapsed
 * against ETA, and the loss, which is the one that says whether the hours are
 * buying anything.
 *
 * The dials are under them for a different reason: two cards training the same
 * set differ only in their dials, and a board where you cannot see which is
 * which is a board of identical rectangles.
 */
export function SessionCard({
  s, ds, onEdit, onStart, onStop, onDelete, onOpenDataset,
}: {
  s: Session
  /** The row from the dataset listing, joined on the page rather than served
   *  with the card: `/api/sessions` is polled and touches no volume, and the
   *  counts are already loaded beside it. Missing means the set was deleted
   *  out from under the card, which the card says rather than hides. */
  ds?: DatasetRow
  onEdit: () => void
  onStart: () => void
  onStop: () => void
  onDelete: () => void
  onOpenDataset: () => void
}) {
  const [asking, setAsking] = useState(false)
  const active = isActive(s)
  const pct = Math.max(0, Math.min(100, Math.round(Number(s.percent ?? 0))))

  // A draft that was saved on the way to making a set is the one card with an
  // instruction rather than a status: it is not waiting on the GPU, it is
  // waiting on you.
  const unfinished = !s.dataset || !s.lora_name || !s.trigger_word

  const meta = [
    s.epoch ? `epoch ${s.epoch}/${s.total_epochs ?? '?'}` : '',
    s.step ? `step ${s.step}/${s.total_steps ?? '?'}` : '',
    s.rate ?? '',
    s.elapsed ? `${s.elapsed}${s.eta ? ` / ${s.eta}` : ''}` : s.eta ? `ETA ${s.eta}` : '',
    typeof s.loss === 'number' ? `loss ${s.loss.toFixed(4)}` : '',
  ].filter(Boolean).join(' · ')

  const p = s.params
  // Rank beside alpha, because the effective strength is alpha ÷ rank — the two
  // numbers only mean anything as a pair, and they used to sit four fields
  // apart with the epoch count between them.
  const dials: [string, string][] = p ? [
    ['rank', String(p.network_dim)],
    ['alpha', String(p.network_alpha)],
    ['batch', String(p.batch_size)],
    ['lr', String(p.learning_rate)],
    ['epochs', String(p.max_train_epochs)],
    ['res', String(p.resolution)],
    ['optimizer', String(p.optimizer_type)],
    ['schedule', String(p.lr_scheduler)],
    ['timesteps', String(p.timestep_sampling)],
    // Only the `shift` sampling reads it. Shown beside a sampling that ignores
    // it, it would be a number on the card that had nothing to do with the run.
    ...(p.timestep_sampling === 'shift'
      ? [['flow shift', String(p.discrete_flow_shift)] as [string, string]] : []),
  ] : []

  return (
    <div className={`sess ${s.status}`} data-session={s.id}>
      <div className="sess-head">
        <b className="sess-name">{s.lora_name || 'Untitled session'}</b>
        <span className={`chip ${s.status}`}>{LABEL[s.status] ?? s.status}</span>
      </div>

      <div className="sess-sub">
        {s.dataset ? (
          // The set is a place, so it is a link to that place rather than a
          // label repeating a name you chose in a menu.
          <button className="link" type="button" onClick={onOpenDataset}
                  title="Open this set in the editor">
            {s.dataset}
          </button>
        ) : <span className="muted">no set yet</span>}
        {s.dataset && (
          <span className="muted">
            {ds ? countLine(ds) : 'set missing'}
          </span>
        )}
        {s.trigger_word && <span className="muted">trigger “{s.trigger_word}”</span>}
      </div>

      {(active || pct > 0) && (
        <>
          <div className="bar"><i style={{ width: `${pct}%` }} /></div>
          <div className="sess-row">
            <span className="muted grow">
              {s.stopping ? 'Stopping — finishing the step it is on'
                : s.phase ? `${s.phase} · ${pct}%` : `${pct}%`}
            </span>
            <span className="muted">{meta}</span>
          </div>
        </>
      )}

      {!!dials.length && (
        <div className="sess-dials">
          {dials.map(([k, v]) => (
            <span key={k}><i>{k}</i>{v}</span>
          ))}
        </div>
      )}

      {(s.note || s.error) && (
        <p className={s.error ? 'sess-note err' : 'sess-note muted'}>{s.error || s.note}</p>
      )}
      {s.status === 'completed' && !!s.files?.length && (
        <p className="sess-note muted">
          {s.files.length} checkpoint{s.files.length === 1 ? '' : 's'}
          {s.duration_s ? ` · ${Math.round(s.duration_s / 60)} min` : ''}
          {s.output_dir ? ` · ${s.output_dir}` : ''}
        </p>
      )}

      <div className="sess-acts">
        {active ? (
          <button className="s" data-act="stop" type="button" disabled={!!s.stopping}
                  onClick={() => setAsking(true)}>
            {s.stopping ? 'Stopping…' : 'Cancel run'}
          </button>
        ) : (
          <>
            <button className="s" data-act="edit" type="button" onClick={onEdit}>
              {unfinished ? 'Continue setup' : 'Edit'}
            </button>
            <button className="s" data-act="delete" type="button"
                    onClick={() => {
                      if (confirm(`Delete the session “${s.lora_name || 'Untitled'}”?\n\n`
                        + 'The card and its setup go. Any checkpoints it already wrote '
                        + 'stay in loras/ and are deleted from Settings.')) onDelete()
                    }}>
              Delete
            </button>
            <button className="b" data-act="start" type="button" disabled={unfinished}
                    onClick={onStart}>
              {s.runs ? 'Run again' : 'Start training'}
            </button>
          </>
        )}
      </div>

      {asking && (
        <StopDialog name={s.lora_name || 'this run'}
                    onStop={() => { setAsking(false); onStop() }}
                    onDelete={() => { setAsking(false); onDelete() }}
                    onClose={() => setAsking(false)} />
      )}
    </div>
  )
}

const LABEL: Record<string, string> = {
  draft: 'Not started',
  queued: 'Queued',
  running: 'Training',
  completed: 'Done',
  stopped: 'Stopped',
  failed: 'Failed',
  unknown: 'Expired',
}

/** Images and clips counted apart. A set of 24 images and 6 clips is not a set
 *  of 30 of anything, and today only one of those two numbers can be trained
 *  on — see the TODO at `train_job`. */
function countLine(d: DatasetRow) {
  const bits = [`${d.count} image${d.count === 1 ? '' : 's'}`]
  if (d.videos) bits.push(`${d.videos} clip${d.videos === 1 ? '' : 's'}`)
  return bits.join(' · ')
}

/**
 * Cancelling is two different acts, and the dialog is where they separate.
 *
 * Stop keeps the card and everything the run wrote: the epochs already saved
 * survive, the bar stays where it got to, and the card can be run again with a
 * dial changed — which is what makes stopping a choice rather than a loss.
 * Delete takes the card away as well. Cancel does nothing at all, which is the
 * one this dialog exists to make available: a stop you cannot take back is a
 * button people learn not to press.
 */
function StopDialog({ name, onStop, onDelete, onClose }: {
  name: string
  onStop: () => void
  onDelete: () => void
  onClose: () => void
}) {
  return (
    <Sheet id="stop-ask" onClose={onClose}>
      <div className="sheet-head">
        <div>
          <h3 style={{ margin: 0 }}>Cancel {name}?</h3>
          <p className="sub" style={{ marginTop: 8, marginBottom: 0 }}>
            {/* No worked example with numbers in it: the card beside this one
                has its own epoch count, and an illustration using someone
                else's reads as a statement about this run. */}
            The GPU stops either way. Every checkpoint written so far stays in
            <code> loras/</code>, so a run stopped part-way is still the epochs
            it got through.
          </p>
        </div>
      </div>
      <div className="sess-acts" style={{ marginTop: 18 }}>
        <button className="s" id="ask-cancel" type="button" onClick={onClose}>
          Keep training
        </button>
        <span className="grow" />
        <button className="s danger" id="ask-delete" type="button" onClick={onDelete}>
          Stop and delete the session
        </button>
        <button className="b" id="ask-stop" type="button" onClick={onStop}>
          Stop, keep the checkpoints
        </button>
      </div>
    </Sheet>
  )
}
