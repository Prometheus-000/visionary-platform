import { useEffect, useState } from 'react'

import { everyMs, failed } from '../api/client'
import { status, stop, thumbUrl, train } from '../api/routes'
import { IconSliders, IconTag, IconTrigger } from '../icons'
import { Editor } from '../datasets/Editor'
import { useDatasets } from '../datasets/useDatasets'
import { useStore } from '../store'

/**
 * Train: the same shape as Generate, for the same reason.
 *
 * Subject in the middle, your library on the right, the console pinned along the bottom.
 * The console is what makes this layout work here — a set of eighty images is a long
 * scroll, and the controls that start the run must not be somewhere down inside it.
 *
 * **Train no longer owns a dataset; it picks one.** The set you are looking at is the set
 * you train, so there is no second place to choose one and nothing that can disagree with
 * the contact sheet.
 */
const DIALS = {
  dim: { label: 'Rank', value: '32', title: 'Network dimension — how much the LoRA can learn. Higher fits more and overfits sooner.' },
  epochs: { label: 'Epochs', value: '30', title: 'Passes over the set. A checkpoint is saved each one, so this is also how many you get to choose between.' },
  lr: { label: 'Learning rate', value: '0.0001', title: 'Step size per update. 1e-4 is the usual starting point for a rank-32 LoRA.' },
  alpha: { label: 'Alpha', value: '32', title: "Scales the LoRA's contribution: the effective strength is alpha ÷ rank. Equal to rank means no scaling." },
  res: { label: 'Resolution', value: '1024', title: 'Training pixels per side. Images are bucketed to it; higher costs VRAM quadratically.' },
  rep: { label: 'Repeats', value: '1', title: 'How many times each image is seen per epoch. Raise it for a set too small to fill an epoch.' },
  bs: { label: 'Batch size', value: '1', title: 'Images per update. Higher is steadier and needs more VRAM.' },
  seed: { label: 'Seed', value: '42', title: 'Fixes shuffling and noise so two runs differing in one dial are actually comparable.' },
} as const

type Dial = keyof typeof DIALS

export function Train({ onLightbox }: { onLightbox: (src: string) => void }) {
  const setTrainPct = useStore((s) => s.setTrainPct)
  const ds = useDatasets()
  const [name, setName] = useState('')
  const [trig, setTrig] = useState('')
  const [dials, setDials] = useState<Record<Dial, string>>(
    Object.fromEntries(Object.entries(DIALS).map(([k, v]) => [k, v.value])) as Record<Dial, string>,
  )
  const [adv, setAdv] = useState(false)
  const [err, setErr] = useState<string | null>(null)
  const [job, setJob] = useState<string | null>(null)
  const [run, setRun] = useState<{
    phase: string; pct: number; meta: string; done?: string; note?: string; dir?: string
    files?: number; minutes?: number
  } | null>(null)

  useEffect(() => {
    void ds.load()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  // Only a saved set lends its name. A draft's handle is `set_3`, a placeholder standing in
  // for a name you have not chosen yet — proposing it as the LoRA's name would turn it into
  // one by default.
  useEffect(() => {
    const row = ds.rows.find((r) => r.name === ds.open)
    if (!row) return
    if (row.trigger_word && !trig) setTrig(row.trigger_word)
    if (row.saved && !name) setName(row.name)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [ds.open, ds.rows])

  const row = ds.rows.find((r) => r.name === ds.open)
  const ready = !!row && row.count > 0 && !!name.trim() && !!trig.trim()
  const hint = !ds.open ? 'Drop images, or pick a set'
    : !row || !row.count ? 'This set is empty'
    : !name.trim() || !trig.trim() ? 'Name it and set a trigger word'
    : ''

  const start = async () => {
    setErr(null)
    const r = await train({
      dataset: ds.open, lora_name: name.trim(), trigger_word: trig.trim(),
      network_dim: dials.dim, network_alpha: dials.alpha, max_train_epochs: dials.epochs,
      learning_rate: dials.lr, resolution: dials.res, num_repeats: dials.rep,
      batch_size: dials.bs, seed: dials.seed,
    })
    if (failed(r)) return setErr(r.error)
    setJob(r.job_id)
    setRun({ phase: 'Starting…', pct: 0, meta: '' })
  }

  // 3000ms, not 400: a training run is hours long, and its own container publishes step,
  // epoch, rate and loss. Polling it at the generate loop's interval would spend the
  // browser's six connections on a progress bar.
  useEffect(() => {
    if (!job) return
    const t = everyMs(async () => {
      const s = await status(job)
      if (failed(s)) return
      const bits = [
        s.step ? `step ${s.step}/${String(s.total_steps ?? '?')}` : '',
        s.epoch ? `epoch ${String(s.epoch)}/${String(s.total_epochs ?? '?')}` : '',
        s.rate ? String(s.rate) : '',
        s.eta ? `ETA ${String(s.eta)}` : '',
        typeof s.loss === 'number' ? `loss ${s.loss.toFixed(4)}` : '',
      ].filter(Boolean)
      setRun({ phase: String(s.phase ?? 'Working…'), pct: Number(s.percent ?? 0), meta: bits.join(' · ') })
      setTrainPct(Number(s.percent ?? 0))
      if (s.status === 'completed' || s.status === 'stopped') {
        clearInterval(t)
        // The door goes back to being a door. A finished run left at 100% would read as
        // one still going, and the result is on the Train side anyway.
        setTrainPct(null)
        setRun({
          phase: s.status === 'stopped' ? 'Stopped' : 'Done', pct: 100, meta: bits.join(' · '),
          done: s.status === 'stopped' ? 'Stopped' : 'Training complete',
          note: String(s.note ?? ''), dir: String(s.output_dir ?? ''),
          files: (s.files as unknown[] | undefined)?.length ?? 0,
          minutes: Math.round(Number(s.duration_s ?? 0) / 60),
        })
      } else if (s.status === 'failed') {
        clearInterval(t)
        setTrainPct(null)
        setRun(null)
        setErr(s.error || 'Training failed')
      }
    }, 3000)
    return () => clearInterval(t)
  }, [job, setTrainPct])

  const dial = (k: Dial, cls = 'opt n') => (
    <div className={cls} data-lb={DIALS[k].label} key={k}>
      <span className="lead">{DIALS[k].label}</span>
      <input type="number" title={DIALS[k].title} value={dials[k]}
             step={k === 'lr' ? 0.00001 : k === 'res' ? 64 : undefined}
             onChange={(e) => setDials((d) => ({ ...d, [k]: e.target.value }))} />
    </div>
  )

  const running = !!run && !run.done

  return (
    <div className="view studio" id="v-train">
      <div className="stage">
        <div className="canvas" id="t-canvas">
          {err && <div className="err-box">{err}</div>}
          <Editor ds={ds} onLightbox={onLightbox} />
        </div>

        {/* Pinned, so eighty images cannot push it off. */}
        <div className="console" id="t-console">
          {!running && (
            <div className="opts" style={{ marginTop: 0 }}>
              {/* Named, not iconed. These are the dials that decide whether a run is worth
                  its hours, and a glyph beside a bare "32" cannot say which of rank,
                  alpha, epochs or batch size you are looking at. An icon is a rebus for a
                  word you already know — it cannot tell you *which* hyperparameter you are
                  looking at. So the tooltip is spent on what the number does. */}
              <div className="opt wide">
                <IconTag />
                <input id="lname" placeholder="LoRA name" spellCheck={false}
                       value={name} onChange={(e) => setName(e.target.value)} />
              </div>
              <div className="opt mid">
                <IconTrigger />
                <input id="ltrig" placeholder="trigger word" spellCheck={false}
                       value={trig} onChange={(e) => setTrig(e.target.value)} />
              </div>
              {dial('dim')}
              {dial('epochs')}
              {/* Not `.n`: the narrow box fits three digits, and 0.0001 is six. A learning
                  rate clipped to "0.000" is worse than no field at all. */}
              {dial('lr', 'opt')}
              <span className="actions">
                <span className="muted" id="train-hint">{hint}</span>
                <button className={`opt ib${adv ? ' on' : ''}`} id="t-toggle-adv" type="button"
                        title="Advanced" onClick={() => setAdv((v) => !v)}>
                  <IconSliders />
                </button>
                <button className="b" id="go-train" type="button" disabled={!ready}
                        onClick={() => void start()}>
                  Start training
                </button>
              </span>
            </div>
          )}

          {adv && !running && (
            <div id="train-adv" className="adv">
              <div className="opts" style={{ marginTop: 0 }}>
                {dial('alpha')}
                {dial('res')}
                {dial('rep')}
                {dial('bs')}
                {dial('seed')}
                <span className="actions"><span className="muted">Krea 2 RAW · bf16</span></span>
              </div>
            </div>
          )}

          {run && (
            <div id="step-run" style={{ marginTop: 11 }}>
              <div className="row">
                <b id="run-phase" className="grow">{run.phase}</b>
                <span className="muted" id="run-pct">{run.pct}%</span>
              </div>
              <div className="bar"><i id="run-bar" style={{ width: `${run.pct}%` }} /></div>
              <div className="row" style={{ marginTop: 9 }}>
                <span className="muted grow" id="run-meta">{run.meta}</span>
                {!run.done && (
                  <button className="s" id="do-stop" type="button"
                          onClick={() => { if (job) void stop(job) }}>
                    Stop &amp; keep checkpoints
                  </button>
                )}
              </div>
              {run.done && (
                <div id="run-done" style={{ marginTop: 11 }}>
                  <b>{run.done}</b>
                  <p className="muted" style={{ marginTop: 7 }}>{run.note}</p>
                  <p className="muted" style={{ marginTop: 7 }}><code>{run.dir}</code></p>
                  <p className="muted" style={{ marginTop: 5 }}>
                    {run.files} checkpoint(s) · {run.minutes} min
                  </p>
                </div>
              )}
            </div>
          )}
        </div>
      </div>

      {/* Every set you already have, always open. The second way in. */}
      <aside className="drawer" id="ds-drawer">
        <div className="drawer-in">
          <div className="drawer-head">
            <span className="grow" />
            {/* Back to the drop target. Nothing is destroyed — the set stays in the rail. */}
            <button className="s" id="ds-fresh" type="button" onClick={() => void ds.choose(null)}>
              + New set
            </button>
          </div>
          {ds.error && <div className="err-box">{ds.error}</div>}
          <div id="ds-list" className="grid">
            <Rail ds={ds} />
          </div>
        </div>
      </aside>
    </div>
  )
}

/** Drafts first: they are the set you are working on, and the reason they are labelled at
 *  all is that the label is a promise about what happens to them. */
function Rail({ ds }: { ds: ReturnType<typeof useDatasets> }) {
  const drafts = ds.rows.filter((d) => !d.saved)
  const saved = ds.rows.filter((d) => d.saved)

  const card = (d: (typeof ds.rows)[number]) => (
    <div className="ds-row" key={d.name}>
      <button className={['ds-card', d.name === ds.open ? 'sel' : '', d.saved ? '' : 'draft']
                .filter(Boolean).join(' ')}
              type="button" onClick={() => void ds.choose(d.name)}>
        {d.cover
          ? <img className="ds-cover" loading="lazy" src={thumbUrl(d.name, d.cover)} alt="" />
          : <div className="ds-cover empty">▤</div>}
        <div className="ds-meta">
          <b>{d.name}</b>
          <div className="muted" style={{ marginTop: 3, fontSize: 12 }}>
            {d.count} image{d.count === 1 ? '' : 's'}
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
