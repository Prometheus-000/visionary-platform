import { useState } from 'react'

import type { Session } from '../api/types'
import type { DatasetRow } from '../datasets/useDatasets'
import { ago } from '../gallery/types'
import { Sheet } from '../ui/Sheet'
import { useBusy } from '../ui/useBusy'
import { isActive } from './useSessions'

/**
 * One training session, as a card.
 *
 * The card is what replaced the full-page run console, and the reason is the
 * backend rather than the layout: `train_job` never shared anything between
 * runs, so "one at a time" was a property of a page that could only hold one
 * `job` in a variable. A board of cards is what the backend was always able to
 * do.
 *
 * **The card is portrait, because a run is read down rather than along.** It
 * was a full-width row first, and the fault was not density — it was that every
 * fact sat on its own horizontal band, so telling two runs apart meant scanning
 * 1180px and then scanning it again one row down. A 3:4 card puts the whole run
 * inside one fixation, and four of them side by side are compared by looking
 * rather than by tracking.
 *
 * **What a card says is what the terminal says.** A bar alone cannot tell
 * training from stuck, and the numbers that tell them apart are the ones tqdm
 * prints: step over total, epoch over total, the rate — `it/s` or `s/it`,
 * whichever way up it came, because which one it is is information — elapsed
 * against ETA, and the loss, which is the one that says whether the hours are
 * buying anything.
 *
 * **The whole card is the way in.** Edit, and Start/Run again, were two buttons
 * saying one thing: open this run's settings. Nobody re-runs a finished LoRA
 * unchanged — you come back to it to move a dial — so the press that mattered
 * was always the one that opens the form, and the form already carries Start.
 * A click anywhere on the card is that press, which is why the resting card has
 * no button on it at all.
 */
export function SessionCard({
  s, ds, onEdit, onStop, onDelete, onOpenDataset,
}: {
  s: Session
  /** The row from the dataset listing, joined on the page rather than served
   *  with the card: `/api/sessions` is polled and touches no volume, and the
   *  counts are already loaded beside it. Missing means the set was deleted
   *  out from under the card, which the card says rather than hides. */
  ds?: DatasetRow
  onEdit: () => void
  /** These two return the board's promise rather than firing and forgetting,
   *  and the promise is the point: `useSessions.act` awaits the mutation *and*
   *  the reload behind it, so it does not settle until the card this press is
   *  about has actually gone or changed state. A handler typed `() => void`
   *  could only be timed by guessing. */
  onStop: () => void | Promise<unknown>
  onDelete: () => void | Promise<unknown>
  onOpenDataset: () => void
}) {
  const [asking, setAsking] = useState(false)
  // Delete is a request plus a full board reload, which on a cold API is well
  // over a second of a card sitting there looking untouched. The ✕ stayed
  // enabled through all of it, so the second press re-opened the confirm dialog
  // for a session already on its way out.
  const { busy, run } = useBusy()
  const active = isActive(s)
  const pct = Math.max(0, Math.min(100, Math.round(Number(s.percent ?? 0))))

  // A draft that was saved on the way to making a set is the one card with an
  // instruction rather than a status: it is not waiting on the GPU, it is
  // waiting on you.
  const unfinished = !s.dataset || !s.lora_name || !s.trigger_word

  // **A finished run draws no bar.** A bar answers "how far along", and a
  // completed run has already answered it — so what was left was a full-width
  // solid rule sitting between the name and the numbers, which is a divider,
  // and it read as one. Stopped and failed keep theirs, because there the
  // fraction is the whole point: it is where the run got to.
  const showBar = active || (pct > 0 && s.status !== 'completed')

  // tqdm's line, as pairs rather than a `·`-joined sentence. In a portrait card
  // a single line wraps to three ragged ones and the numbers stop lining up
  // between cards, which is the only reason to show them on a board.
  const stats: [string, string][] = [
    ['epoch', s.epoch ? `${s.epoch}/${s.total_epochs ?? '?'}` : ''],
    ['step', s.step ? `${s.step}/${s.total_steps ?? '?'}` : ''],
    ['rate', s.rate ?? ''],
    ['time', s.elapsed ? `${s.elapsed}${s.eta ? ` / ${s.eta}` : ''}` : s.eta ? `ETA ${s.eta}` : ''],
    ['loss', typeof s.loss === 'number' ? s.loss.toFixed(4) : ''],
  ].filter(([, v]) => v) as [string, string][]

  const p = s.params
  // Rank beside alpha, because the effective strength is alpha ÷ rank — the two
  // numbers only mean anything as a pair.
  //
  // The optimizer, the scheduler, the timestep sampling and the flow shift used
  // to be here and are not: they are the four dials nobody varies between two
  // runs of the same set, so on a board they were four columns of the same word
  // repeated down every card. They are all still in the form, one click away,
  // which is where a value you set once belongs.
  const dials: [string, string][] = p ? [
    ['rank', String(p.network_dim)],
    ['alpha', String(p.network_alpha)],
    ['batch', String(p.batch_size)],
    ['lr', String(p.learning_rate)],
    ['epochs', String(p.max_train_epochs)],
    ['res', String(p.resolution)],
  ] : []

  return (
    <div className={`sess ${s.status}`} data-session={s.id}>
      {/* A real button rather than an onClick on the card, so it is focusable,
          answers Enter and Space, and names itself to a screen reader. It sits
          under the content and the content is click-through — see `.sess-hit`
          in the stylesheet for why that is the arrangement rather than a
          wrapper, which cannot contain the buttons that are still on the card. */}
      <button className="sess-hit" type="button" onClick={onEdit} disabled={!!busy}
              aria-label={`Open the settings for ${s.lora_name || 'this session'}`} />

      <div className="sess-head">
        <span className={`chip ${s.status}`}>{LABEL[s.status] ?? s.status}</span>
        {/* Two DONE cards are otherwise indistinguishable in time, and "which
            was the recent k3nan" is the question a board of finished runs gets
            asked most. `updated` over `created` because a re-run moves it. */}
        {!active && (
          <span className="muted sess-when">{ago(s.updated ?? s.created)}</span>
        )}
        <span className="grow" />
        {!active && (
          // The same corner and the same gesture as a set card's delete. It is
          // the one thing on a resting card that is not "open this", so it is
          // the one thing that gets its own target — and it appears on hover,
          // because a board at rest should carry no destructive control.
          //
          // It says ✕ at rest and … while the delete and the reload behind it
          // are in flight. Staying visible through that is the stylesheet's job
          // now — `.sess-x:disabled` holds it at full opacity, because `.sess-x`
          // is otherwise opacity:0 until the card is hovered, which is right for
          // a resting board and wrong for the second or two a delete takes: the
          // hand leaves the card as soon as the confirm dialog is dismissed and
          // would take the only sign that anything is happening with it.
          <button className="sess-x" type="button" disabled={!!busy}
                  title={busy === 'del' ? 'Deleting…' : 'Delete this session'}
                  aria-label={busy === 'del' ? 'Deleting this session'
                    : `Delete the session ${s.lora_name || 'Untitled'}`}
                  onClick={() => {
                    if (!confirm(`Delete the session “${s.lora_name || 'Untitled'}”?\n\n`
                      + 'The card and its setup go. Any LoRA files it already wrote '
                      + 'stay in loras/ — delete those from Settings.')) return
                    void run('del', async () => { await onDelete() })
                  }}>{busy === 'del' ? '…' : '×'}</button>
        )}
      </div>

      <b className="sess-name">{s.lora_name || 'Untitled session'}</b>

      <div className="sess-sub">
        {s.dataset ? (
          // The set is a place, so it is a link to that place rather than a
          // label repeating a name you chose in a menu.
          <button className="link" type="button"
                  onClick={onOpenDataset}
                  title="Open this set in the editor">
            {s.dataset}
          </button>
        ) : <span className="muted">no set yet</span>}
        {s.dataset && <span className="muted">{ds ? countLine(ds) : 'set missing'}</span>}
        {s.trigger_word && <span className="muted">trigger “{s.trigger_word}”</span>}
      </div>

      <span className="grow" />

      {showBar && (
        <div className="sess-prog">
          <div className="bar"><i style={{ width: `${pct}%` }} /></div>
          <div className="sess-pct">
            <span className="muted grow">
              {s.stopping ? 'Stopping — finishing the step it is on'
                : s.phase ? s.phase : LABEL[s.status] ?? ''}
            </span>
            <span>{pct}%</span>
          </div>
        </div>
      )}

      {!!stats.length && (
        <div className="sess-live">
          {stats.map(([k, v]) => <span key={k}><i>{k}</i>{v}</span>)}
        </div>
      )}

      {/* "3 checkpoints" on a finished session named the output after the
          mechanism that wrote it, and the LoRA picker two screens away calls the
          same files LoRAs. They are one LoRA saved at three epochs — three files
          you choose between — so the count is of files and the noun is the one
          the rest of the app uses. "Checkpoint" is left to the stop dialog,
          where it means the other thing: an epoch snapshot of a session that
          never reached its finished LoRA at all. */}
      {s.status === 'completed' && !!s.files?.length && (
        <p className="sess-note muted">
          {s.files.length} LoRA file{s.files.length === 1 ? '' : 's'}
          {s.duration_s ? ` · ${Math.round(s.duration_s / 60)} min` : ''}
        </p>
      )}

      {!!dials.length && (
        <div className="sess-dials">
          {dials.map(([k, v]) => <span key={k}><i>{k}</i>{v}</span>)}
        </div>
      )}

      {(s.note || s.error) && (
        <p className={s.error ? 'sess-note err' : 'sess-note muted'}>{s.error || s.note}</p>
      )}

      <div className="sess-acts">
        {active ? (
          // "Cancel run" was the board's second noun for the thing every other
          // line on this card calls a session. It pairs with "Keep training" in
          // the dialog it opens, which is the sentence it is really half of.
          <button className="t danger" data-act="stop" type="button" disabled={!!s.stopping}
                  onClick={() => setAsking(true)}>
            {s.stopping ? 'Stopping…' : 'Cancel training'}
          </button>
        ) : unfinished ? (
          // Not a button: the card already is one, and a card whose whole face
          // opens the form does not need a second thing to press inside it.
          // What is missing is a sentence, so it is written as one.
          <span className="muted sess-todo">Finish setup →</span>
        ) : null}
      </div>

      {/* The dialog stays up until the request and the reload behind it have
          landed, and closes itself after. Closing first left nothing on screen
          saying anything had happened at all: the card goes on reporting
          Training until a later poll agrees it stopped, and the button that was
          pressed went away with the sheet. Keep training is shut for the same
          window — a stop already sent is not one this dialog can call off. */}
      {asking && (
        <StopDialog name={s.lora_name || 'this session'} busy={busy}
                    onStop={() => void run('stop', async () => {
                      await onStop(); setAsking(false)
                    })}
                    onDelete={() => void run('del', async () => {
                      await onDelete(); setAsking(false)
                    })}
                    onClose={() => { if (!busy) setAsking(false) }} />
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
 * Stop keeps the card and everything the session wrote: the epochs already saved
 * survive, the bar stays where it got to, and the card can be started again with
 * a dial changed — which is what makes stopping a choice rather than a loss.
 * Delete takes the card away as well. Cancel does nothing at all, which is the
 * one this dialog exists to make available: a stop you cannot take back is a
 * button people learn not to press.
 *
 * **This is the one screen that says "checkpoint", and it is the one screen
 * where the word means something a LoRA does not.** Everywhere else the two were
 * used for the same file, so "checkpoint" was a second name for the artifact
 * rather than a name for a different thing. Here it is a different thing: what a
 * stopped session leaves behind is the LoRA as it stood at some epoch, and never
 * the finished one — so the sentence below spends a clause teaching exactly that,
 * which is what earns the word its place on the button under it.
 */
function StopDialog({ name, busy, onStop, onDelete, onClose }: {
  name: string
  /** Which of the two outcomes is in flight, or null. The pressed button says
   *  what it is doing and its neighbours go quiet — including Keep training,
   *  because a stop already sent is not one this dialog can call off. */
  busy: string | null
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
                else's reads as a statement about this session. */}
            The GPU stops either way. Every epoch saved so far stays in
            <code> loras/</code> as a checkpoint — the LoRA as it was at that
            epoch, not the finished one — so a session stopped part-way is still
            the epochs it got through.
          </p>
        </div>
      </div>
      <div className="sess-acts" style={{ marginTop: 18 }}>
        <button className="t" id="ask-cancel" type="button" disabled={!!busy}
                onClick={onClose}>
          Keep training
        </button>
        <span className="grow" />
        <button className="t danger" id="ask-delete" type="button" disabled={!!busy}
                onClick={onDelete}>
          {busy === 'del' ? 'Deleting…' : 'Stop and delete the session'}
        </button>
        <button className="b" id="ask-stop" type="button" disabled={!!busy}
                onClick={onStop}>
          {busy === 'stop' ? 'Stopping…' : 'Stop, keep the checkpoints'}
        </button>
      </div>
    </Sheet>
  )
}
