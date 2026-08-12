import { useMemo, useState } from 'react'

import { failed } from '../api/client'
import {
  deleteLora, downloadFamily, setToken, startDownload, startGdrive,
} from '../api/routes'
import type { AppState, GpuChoice, LoraEntry, ModelEntry } from '../api/types'
import { useStore } from '../store'
import { fmtBytes } from '../format'
import { IconClose } from '../icons'
import { useDownload } from './useDownload'

/**
 * Everything decided once, behind the gear.
 *
 * Nothing downloads on its own — weights are chosen explicitly, here. The GPU
 * pickers live here for the same reason: a card is set per session and confirms
 * a cold start when it changes, so it was 71px of composer for a decision no
 * take varies by.
 */
export function Settings({
  state, open, onClose, onReload,
}: {
  state: AppState | null
  open: boolean
  onClose: () => void
  onReload: () => void
}) {
  const [token, setTokenValue] = useState('')
  const [tokenNote, setTokenNote] = useState<string | null>(null)
  const [driveUrl, setDriveUrl] = useState('')
  const [driveFolder, setDriveFolder] = useState('')
  const [loraError, setLoraError] = useState<string | null>(null)
  const dl = useDownload()

  // Grouped by family, in catalogue order. Twenty-odd flat cards is a wall you
  // scroll rather than a list you read, and the group is the unit you actually
  // decide in: you want the Wan stack or you do not.
  const families = useMemo(() => {
    const out: { name: string; items: ModelEntry[] }[] = []
    for (const m of state?.models ?? []) {
      const g = out.find((f) => f.name === m.family)
      if (g) g.items.push(m)
      else out.push({ name: m.family, items: [m] })
    }
    return out
  }, [state])

  if (!open) return null

  const saveToken = async () => {
    const r = await setToken(token)
    setTokenNote(failed(r) ? r.error : 'Token saved.')
    setTokenValue('')
    if (!failed(r)) onReload()
  }

  const removeLora = async (l: LoraEntry) => {
    // The dialog is the entire safety net: the route unlinks and there is
    // nothing behind it. So it says how much is going, and whether it can come
    // back — two different sentences, because a Wan speed pair is a download
    // and a LoRA you trained is however many hours that run took.
    const n = l.files.length
    const ok = confirm(
      `Permanently delete “${l.name}”?\n\n` +
      `${n} file${n === 1 ? '' : 's'} (${fmtBytes(l.bytes)}) unlinked from the volume.\n` +
      (l.catalogue
        ? `It is part of ${l.catalogue} in the catalogue below, so it can be downloaded again.`
        : 'This cannot be undone.'),
    )
    if (!ok) return
    const r = await deleteLora(l.root)
    if (failed(r)) return setLoraError(r.error)
    setLoraError(null)
    // The whole sheet, not just this list: deleting a catalogue LoRA moves it
    // back to "missing" in the cards below.
    onReload()
  }

  const loras = state?.loras ?? []
  const loraBytes = loras.reduce((a, l) => a + (Number(l.bytes) || 0), 0)

  return (
    <div id="settings" className="scrim">
      <div className="sheet">
        <div className="sheet-head">
          <h1 className="grow">Settings</h1>
          <button className="ico" id="settings-x" type="button" onClick={onClose}>
            <IconClose />
          </button>
        </div>

        <div className="card">
          <label>GPU</label>
          <div className="row" style={{ gap: 10 }}>
            <GpuSelect id="g-gpu" label="Images" side="image" spec={state?.gpus.image} />
            <GpuSelect id="v-gpu" label="Video" side="video" spec={state?.gpus.video} />
          </div>
          <p className="muted" style={{ margin: '9px 2px 0' }}>
            Changing a card costs one cold start while the model loads. Runs after it are warm.
          </p>
        </div>

        {/* The token, and only the token. "Download missing" used to sit in
            this row, which put the one button that pulls the entire catalogue
            next to a password field it has nothing to do with. */}
        <div className="card">
          <label>HuggingFace token</label>
          <div className="row">
            <input id="tok" type="password" className="grow" placeholder="hf_…"
                   autoComplete="off" value={token}
                   onChange={(e) => setTokenValue(e.target.value)} />
            <button className="s" type="button" onClick={() => void saveToken()}>Save</button>
          </div>
          <p className="muted" style={{ marginTop: 8 }}>
            Needed for Krea 2 RAW and Turbo, which are gated. Accept the licence at
            huggingface.co/krea/Krea-2-Raw with the same account.{' '}
            <span id="tok-state">
              {tokenNote ? <span className="ok">{tokenNote}</span>
                : state?.hf_token_set ? <span className="ok">Token saved.</span>
                : <span className="warn">No token saved.</span>}
            </span>
          </p>
        </div>

        {/* The other way weights arrive. Most LoRAs worth having were never
            published to HuggingFace — they are a link someone sent you. Same
            card shape as the token, deliberately: another place weights come
            from, not another kind of thing. */}
        <div className="card">
          <label>Google Drive</label>
          <div className="row">
            <input id="gd-url" className="grow" placeholder="Drive link or file id"
                   autoComplete="off" spellCheck={false} value={driveUrl}
                   onChange={(e) => setDriveUrl(e.target.value)} />
            <input id="gd-folder" placeholder="folder (optional)" autoComplete="off"
                   spellCheck={false} style={{ width: 158, flex: 'none' }}
                   title="Group the files under loras/{name}/ — for a matched pair that belongs together. Leave blank to drop them in loose."
                   value={driveFolder} onChange={(e) => setDriveFolder(e.target.value)} />
            <button className="b" type="button" disabled={dl.busy}
                    style={{ padding: '9px 16px', fontSize: 13 }}
                    onClick={() => {
                      if (!driveUrl.trim()) return
                      void dl.begin('gdrive', 'dl_gdrive', 'Downloaded.',
                        () => startGdrive(driveUrl.trim(), driveFolder.trim() || undefined),
                        onReload)
                    }}>
              Download
            </button>
          </div>
          <p className="muted" style={{ marginTop: 8 }}>
            Lands in <code>loras/</code>, ready to name in a prompt. Only{' '}
            <code>.safetensors</code> is kept — a folder's preview images and readme are left
            behind. The link has to be shared with anyone who has it.
          </p>
          {/* No progress bar, unlike the cards below. Drive does not say how big
              a file is before it sends it, so a bar here could only sit at zero
              for the length of the transfer — which is what "stuck" looks like.
              The byte count and the rate move, and moving is the whole job. */}
          <Line p={dl.progressOf('gdrive')}
                onCancel={() => void dl.cancel('gdrive', 'dl_gdrive')} />
        </div>

        <div className="card">
          <div className="row" style={{ alignItems: 'baseline', marginBottom: 4 }}>
            <label className="grow" style={{ margin: 0 }}>LoRAs</label>
            <span className="muted" id="lora-total">
              {loras.length ? `${loras.length} · ${fmtBytes(loraBytes)}` : ''}
            </span>
          </div>
          {loraError && <div style={{ marginTop: 10 }}><div className="err-box">{loraError}</div></div>}
          <div id="lora-list">
            {loras.length ? loras.map((l) => (
              <div className="lora-row" key={l.root}>
                <div className="grow" style={{ minWidth: 0 }}>
                  <b>{l.name}</b>
                  {l.trigger_word ? <> <code>{l.trigger_word}</code></> : null}
                  <div className="muted">
                    {l.files.length} file{l.files.length === 1 ? '' : 's'} · {fmtBytes(l.bytes)}
                    {l.catalogue ? ` · ${l.catalogue}` : ''}
                  </div>
                </div>
                <button className="lora-x" title="Delete" type="button"
                        onClick={() => void removeLora(l)}>✕</button>
              </div>
            )) : (
              // Both ways in are directly above this card, so the empty state
              // points at them rather than saying "nothing here".
              <p className="muted" style={{ margin: '2px 0 0' }}>
                Nothing in <code>loras/</code> yet — train one, or paste a Drive link above.
              </p>
            )}
          </div>
        </div>

        <div id="models">
          {families.map((f) => {
            const left = f.items.filter((m) => !m.present)
            const size = left.reduce((a, m) => a + m.approx_gb, 0)
            const id = `fam:${f.name}`
            return (
              <div className="fam" key={f.name}>
                <div className="fam-head">
                  <b>{f.name}</b>
                  <span className="muted">
                    {left.length ? `${left.length} missing · ${size.toFixed(1)} GB` : 'complete'}
                  </span>
                  {/* One short shows no button at all, because that is what its
                      own Download already is. */}
                  {left.length > 1 && (
                    <button className="s" type="button" disabled={dl.busy}
                            onClick={() => void dl.begin(id, id, `${f.name} downloaded.`,
                              () => downloadFamily(f.name, token), onReload)}>
                      Download all {left.length}
                    </button>
                  )}
                </div>
                <Line p={dl.progressOf(id)} bar onCancel={() => void dl.cancel(id, id)} />

                {f.items.map((m) => {
                  const p = dl.progressOf(m.key)
                  return (
                    <div className="card row" key={m.key}>
                      <div className="grow">
                        <b>{m.label}</b> <span className="muted">{m.note}</span>
                        {m.gated && <span className="warn" style={{ fontSize: 12 }}> · gated</span>}
                        <div className="muted" style={{ marginTop: 3 }}><code>{m.repo_id}</code></div>
                      </div>
                      <div style={{ textAlign: 'right' }}>
                        {m.present ? (
                          <span className="ok">✓ {m.size_gb} GB</span>
                        ) : (
                          <>
                            <button className="s" type="button" disabled={dl.busy}
                                    onClick={() => void dl.begin(m.key, `dl_${m.key}`, 'Done',
                                      () => startDownload(m.key), onReload)}>
                              Download {m.approx_gb} GB
                            </button>
                            {p.running && (
                              <button className="s" type="button"
                                      onClick={() => void dl.cancel(m.key, `dl_${m.key}`)}>
                                Cancel
                              </button>
                            )}
                          </>
                        )}
                        <div className={`muted dl-state${p.tone === 'ok' ? ' ok' : p.tone === 'err' ? ' err' : ''}`}>
                          {p.message}
                        </div>
                      </div>
                    </div>
                  )
                })}
              </div>
            )
          })}
        </div>
      </div>
    </div>
  )
}

/**
 * One warning, once per card, and only when you actually change it.
 *
 * Switching starts a container that does not exist yet — on the video side that is
 * 42.5 GB of weights, so the cost is worth a sentence before it is spent rather than a
 * progress bar that sits still for minutes afterwards. Declining puts the select back,
 * because a confirm that leaves the control showing the answer you refused is a control
 * that lies about what the next run will use.
 *
 * The choice is store state and not just a DOM value: `imageBody` and `videoBody` read
 * `gpu` off the store, so a select nothing wrote to would send the deployment's default
 * on every run no matter what this said.
 */
function GpuSelect({ id, label, side, spec }: {
  id: string
  label: string
  side: 'image' | 'video'
  spec: GpuChoice | undefined
}) {
  const value = useStore((s) => s.gpu[side])
  const setGpu = useStore((s) => s.setGpu)
  return (
    <div className="opt" data-lb={label}>
      <span className="lead">{label}</span>
      <select id={id} value={value || spec?.default || ''}
              onChange={(e) => {
                const next = e.target.value
                if (next === spec?.default
                    || confirm(`Switch to ${next}?\n\nThis card has no warm container, `
                      + 'so the next run pays a cold start while the model loads. '
                      + 'Runs after it are warm.')) {
                  setGpu({ [side]: next })
                }
              }}>
        {(spec?.options ?? []).map((o) => <option key={o}>{o}</option>)}
      </select>
    </div>
  )
}

function Line({ p, bar, onCancel }: {
  p: { percent: number; message: string; tone: string; running: boolean }
  bar?: boolean
  onCancel: () => void
}) {
  if (!p.message) return null
  return (
    <div className="fam-prog">
      {bar && <div className="bar"><i style={{ width: `${p.percent}%` }} /></div>}
      <div className="row" style={{ gap: 10, marginTop: bar ? 7 : 0 }}>
        <p className={`grow${p.tone === 'ok' ? ' ok' : p.tone === 'err' ? ' err' : ' muted'}`}
           style={{ margin: 0 }}>
          {p.message}
        </p>
        {p.running && <button className="s" type="button" onClick={onCancel}>Cancel</button>}
      </div>
    </div>
  )
}
