import { useState } from 'react'

import type { AppState, TrainParams } from '../api/types'
import type { DatasetRow } from '../datasets/useDatasets'
import { IconTag, IconTrigger } from '../icons'
import { ErrorNote } from '../ui/ErrorNote'
import { Sheet } from '../ui/Sheet'
import type { Draft } from './useSessions'
import { paramsToForm } from './useSessions'

/**
 * Setting up a session, in a modal.
 *
 * The same controls the console had, in the same `.opt` boxes — the rehousing
 * was deliberate: what changed is that a session is now a card you make, so the
 * form is the moment you make one instead of a strip that is on screen whether
 * or not you are starting anything.
 *
 * **Four fields on open, not eighteen.** Rehousing the strip verbatim also
 * rehoused its shape, and that shape was `train_defaults`: every key the backend
 * serialises, rendered as a control, in the order it happens to send them. So
 * the form's subject was the API schema rather than the task — the README
 * promises you point the trainer at a folder of images, and the sheet asked you
 * to have an opinion about `blocks_to_swap` before it would let you past. What
 * is on open now is what starting a session actually takes: the name, the
 * trigger, the set, and how long to train for. Nothing was removed and no
 * default moved — the other ten dials, the three menus and fp8 are all still
 * here at the same values, one press of More dials away. The difference is only
 * what you have to walk past to start.
 *
 * **Every numeric field carries its name.** This is the one place the "a
 * control that shows its own value gets no label" rule was pushed past what it
 * can carry: "32" is a rank, an alpha, an epoch count or a seed with equal
 * plausibility, and someone who has trained these models for five years had to
 * hover every field to find out which. The tooltip is spent on what the number
 * *does*, which is what a name cannot say.
 *
 * The three menus need no such label — "AdamW 8-bit", "Cosine", "Shift" each
 * name themselves — so what they carry is the note the server sends with them,
 * because what an optimizer costs in VRAM is not inferable from its name.
 */
const DIALS = {
  // Rank beside alpha, and not for tidiness: the effective strength is
  // alpha ÷ rank, so the two are one decision and were four fields apart.
  network_dim: { label: 'Rank', title: 'Network dimension — how much the LoRA can learn. Higher fits more and overfits sooner.' },
  network_alpha: { label: 'Alpha', title: "Scales the LoRA's contribution: the effective strength is alpha ÷ rank. Equal to rank means no scaling." },
  max_train_epochs: { label: 'Epochs', title: 'Passes over the set. A LoRA is written every “Save every” epochs — under More dials — so this divided by that is how many you get to choose between.' },
  learning_rate: { label: 'Learning rate', title: 'Step size per update. 1e-4 is the usual starting point for a rank-32 LoRA.', step: 0.00001 },
  batch_size: { label: 'Batch size', title: 'Images per update. Higher is steadier and needs more VRAM.' },
  resolution: { label: 'Resolution', title: 'Training pixels per side. Images are bucketed to it; higher costs VRAM quadratically.', step: 64 },
  num_repeats: { label: 'Repeats', title: 'How many times each image is seen per epoch. Raise it for a set too small to fill an epoch.' },
  seed: { label: 'Seed', title: 'Fixes shuffling and noise so two sessions differing in one dial are actually comparable.' },
  save_every_n_epochs: { label: 'Save every', title: 'Epochs between saved LoRAs. Every 5 keeps a 30-epoch session to six files instead of thirty; drop to 1 to keep everything a stopped session has written.' },
  discrete_flow_shift: { label: 'Flow shift', title: 'Weights the flow-matching schedule toward the noisy end. Only the Shift timestep sampling reads it.', step: 0.05 },
  blocks_to_swap: { label: 'Blocks to swap', title: 'Offload this many transformer blocks to CPU. Buys VRAM at the cost of speed; 0 keeps everything on the card.' },
} as const

type Dial = keyof typeof DIALS

/**
 * The one dial the task has an opinion about, and the ten the schema has.
 *
 * The split used to be frequency: six you change per session against four you
 * change once a year. That still opened the sheet with rank, alpha, the learning
 * rate and the resolution in front of someone whose entire question was how long
 * to train for — four numbers that have a good default and no local answer.
 * Epochs is the one that has no default worth trusting, because it is the one
 * that decides what the hours cost; the rest come from the server already set.
 *
 * `discrete_flow_shift` is in neither list and that is not an omission: it is
 * disabled unless the timestep sampling is Shift, so it renders beside that menu
 * rather than adrift in a row of free-standing numbers. Rank leads ADVANCED with
 * alpha behind it, because the effective strength is alpha ÷ rank and the two
 * are one decision — the same adjacency `check_train.py` pins.
 */
const PRIMARY: Dial[] = ['max_train_epochs']
const ADVANCED: Dial[] = ['network_dim', 'network_alpha', 'learning_rate', 'batch_size',
                          'resolution', 'num_repeats', 'seed', 'save_every_n_epochs',
                          'blocks_to_swap']

export function SessionForm({
  initial, state, datasets, saving, error, onSave, onClose, onNewDataset,
}: {
  initial: Draft
  state: AppState | null
  datasets: DatasetRow[]
  saving: boolean
  error: string | null
  onSave: (d: Draft, start: boolean) => void
  onClose: () => void
  /** Picking "a new set" saves the card unfinished and walks you to the drop
   *  target. The card is what you come back to — an inactive session with
   *  Continue setup on it — because a form abandoned mid-way is otherwise a
   *  form you retype. */
  onNewDataset: (d: Draft) => void
}) {
  const defaults = state?.train_defaults
  const [name, setName] = useState(initial.lora_name)
  const [trig, setTrig] = useState(initial.trigger_word)
  const [dataset, setDataset] = useState(initial.dataset)
  const [params, setParams] = useState<Record<string, string | boolean>>(
    Object.keys(initial.params).length
      ? initial.params
      : paramsToForm(undefined, (defaults ?? {}) as TrainParams),
  )
  const [adv, setAdv] = useState(false)

  const draft = (): Draft => ({
    id: initial.id, lora_name: name.trim(), trigger_word: trig.trim(), dataset, params,
  })
  const set = (k: string, v: string | boolean) => setParams((d) => ({ ...d, [k]: v }))
  const row = datasets.find((d) => d.name === dataset)
  // The three refusals `/api/sessions/{id}/start` makes, said before the press
  // rather than after it. Saving has no such gate: an unfinished card is the
  // normal state of one made on the way to building a set.
  //
  // Every field named here is one of the four that are always on screen, and
  // that has to stay true now that ten dials sit behind More dials: a footer
  // reading "Pick a set" while the set picker is folded away is a button
  // explaining itself with a control the reader cannot see. Nothing under the
  // disclosure can block a start — the server takes all ten as they come.
  const hint = !name.trim() ? 'Name the LoRA'
    : !trig.trim() ? 'Set a trigger word'
    : !dataset ? 'Pick a set'
    : row && !row.count ? 'That set has no images'
    : ''

  // Not `.n` for the learning rate: the narrow box fits three digits and 0.0001
  // is six, and a rate clipped to "0.000" is worse than no field at all.
  const dial = (k: Dial) => (
    <div className={k === 'learning_rate' ? 'opt' : 'opt n'} data-lb={DIALS[k].label} key={k}>
      <span className="lead">{DIALS[k].label}</span>
      {/* `a-{key}`, matching the ids every check addresses these by — a control
          a check cannot find reads as a feature that is not there. */}
      <input id={`a-${SHORT[k]}`} type="number" title={DIALS[k].title}
             step={'step' in DIALS[k] ? (DIALS[k] as { step: number }).step : undefined}
             disabled={k === 'discrete_flow_shift' && params.timestep_sampling !== 'shift'}
             value={String(params[k] ?? '')}
             onChange={(e) => set(k, e.target.value)} />
    </div>
  )

  const menu = (k: 'optimizer_type' | 'lr_scheduler' | 'timestep_sampling',
                options: { key: string; label: string; note: string }[]) => (
    <div className="opt" key={k}>
      <select id={`a-${SHORT[k]}`} value={String(params[k] ?? '')}
              title={options.find((o) => o.key === params[k])?.note ?? ''}
              onChange={(e) => set(k, e.target.value)}>
        {options.map((o) => <option key={o.key} value={o.key}>{o.label}</option>)}
      </select>
    </div>
  )

  return (
    <Sheet id="sess-form" onClose={onClose}>
      <div className="sheet-head">
        <div className="grow">
          <h3 style={{ margin: 0 }}>{initial.id ? 'Edit session' : 'New training session'}</h3>
          <p className="sub" style={{ margin: '8px 0 0' }}>
            Krea 2 RAW · bf16. This session gets its own container, so it does not
            wait on anything else you have going.
          </p>
        </div>
      </div>

      <ErrorNote err={error} />

      <div className="opts">
        <div className="opt wide">
          <IconTag />
          <input id="lname" placeholder="LoRA name" spellCheck={false} autoFocus
                 value={name} onChange={(e) => setName(e.target.value)} />
        </div>
        <div className="opt mid">
          <IconTrigger />
          <input id="ltrig" placeholder="trigger word" spellCheck={false}
                 value={trig} onChange={(e) => setTrig(e.target.value)} />
        </div>
      </div>

      <div className="opts">
        <div className="opt wide">
          {/* One menu, both ways in. Making a set is not a different kind of
              decision from choosing one — it is the same decision when the set
              does not exist yet — so it is the last option rather than a second
              button beside the field. */}
          <select id="sess-dataset" value={dataset}
                  onChange={(e) => {
                    if (e.target.value === NEW) return onNewDataset({ ...draft(), dataset: '' })
                    setDataset(e.target.value)
                  }}>
            <option value="">Pick a set…</option>
            {datasets.map((d) => (
              <option key={d.name} value={d.name}>
                {d.name} — {d.count} image{d.count === 1 ? '' : 's'}
                {d.videos ? `, ${d.videos} clip${d.videos === 1 ? '' : 's'}` : ''}
                {d.saved ? '' : ' (unsaved)'}
              </option>
            ))}
            <option value={NEW}>+ New set…</option>
          </select>
        </div>
        {!!row?.uncaptioned && (
          <span className="muted">
            {row.uncaptioned} uncaptioned — they train on the bare trigger word
          </span>
        )}
      </div>

      <div className="opts">{PRIMARY.map(dial)}</div>

      <div className="opts">
        {/* A disclosure, not an action — so it carries no chrome. There is one
            filled button on this sheet and it is the one that spends a GPU.
            It is also the whole of the schema now, so it is the last thing
            before the actions rather than a footnote after two rows of dials. */}
        <button className={`t${adv ? ' on' : ''}`} id="t-toggle-adv" type="button"
                onClick={() => setAdv((v) => !v)}>
          {adv ? 'Fewer dials' : 'More dials'}
        </button>
      </div>

      {/* Closed by default, and closed again on the next open: `adv` is state on
          a sheet that is unmounted when it closes. Deliberate — a sheet that
          remembers it was expanded is a sheet that is expanded for everyone who
          once went looking for the seed. */}
      {adv && (
        <div id="train-adv" className="adv">
          <div className="opts" style={{ marginTop: 0 }}>
            {ADVANCED.map(dial)}
          </div>

          {/* The three menus and the shift on their own row, as they were when
              they sat outside the disclosure: each is a word rather than a
              number, and mixed into the numbers they read as one long line of
              values with no edges. */}
          <div className="opts">
            {state && menu('optimizer_type', state.train_optimizers)}
            {state && menu('lr_scheduler', state.lr_schedulers)}
            {state && menu('timestep_sampling', state.timestep_samplings)}
            {dial('discrete_flow_shift')}
            <label className="row" style={{ gap: 7, margin: 0, color: '#ddd', fontSize: 13 }}>
              <input type="checkbox" id="a-fp8" style={{ width: 'auto' }}
                     checked={!!params.fp8}
                     onChange={(e) => set('fp8', e.target.checked)} />
              fp8 base
            </label>
          </div>
        </div>
      )}

      <div className="sess-acts" style={{ marginTop: 20 }}>
        <span className="muted grow" id="train-hint">{hint}</span>
        <button className="t" id="sess-save" type="button" disabled={saving}
                onClick={() => onSave(draft(), false)}>
          Save for later
        </button>
        <button className="b" id="go-train" type="button" disabled={saving || !!hint}
                onClick={() => onSave(draft(), true)}>
          Start training
        </button>
      </div>
    </Sheet>
  )
}

const NEW = '__new__'

/** The short ids the console used, kept: `a-dim`, `a-lr`, `a-bs`. They are what
 *  `tools/ui-checks/check_train.py` addresses these fields by, and an id that
 *  moves with a rehousing is a check that silently stops checking. */
const SHORT: Record<Dial | 'optimizer_type' | 'lr_scheduler' | 'timestep_sampling', string> = {
  network_dim: 'dim', network_alpha: 'alpha', max_train_epochs: 'epochs',
  learning_rate: 'lr', batch_size: 'bs', resolution: 'res', num_repeats: 'rep',
  seed: 'seed', save_every_n_epochs: 'save', discrete_flow_shift: 'shift',
  blocks_to_swap: 'swap', optimizer_type: 'opt', lr_scheduler: 'sched',
  timestep_sampling: 'ts',
}
