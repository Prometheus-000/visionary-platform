import { useMemo, useRef, useState, type ReactNode } from 'react'

import { failed, type ApiError } from '../api/client'
import {
  addCaptionModel, deleteCaptionModel, deleteLora, downloadFamily, loraFileUrl, pushLora,
  setToken, startDownload, startGdrive, startHfLora, uploadLoras,
} from '../api/routes'
import type { AppState, GpuChoice, LoraEntry, ModelEntry } from '../api/types'
import { useStore } from '../store'
import { fmtBytes } from '../format'
import { IconClose, IconDownload, IconUpload } from '../icons'
import { ErrorNote } from '../ui/ErrorNote'
import { useBusy } from '../ui/useBusy'
import { useDownload } from './useDownload'

/**
 * Everything decided once, behind the gear.
 *
 * Two groups, and the groups are headings rather than folds. **Runtime** is
 * the GPU: a card is set per session and confirms a cold start when it
 * changes, so it was 71px of composer for a decision no take varies by.
 * **Models** is the weights — LoRAs, caption models, checkpoints — each a
 * fold, because this is the one screen that has to be scrolled to be used and
 * a catalogue you are not shopping in is twenty rows between you and the LoRA
 * you came to delete. LoRAs open by default (the list you come here for);
 * Checkpoints open while anything is missing and fold once the volume is
 * complete; Caption models fold, because the menu is read on the caption
 * row and edited here about once.
 *
 * Nothing downloads on its own — weights are chosen explicitly, here. The
 * ways a LoRA arrives (Drive, HuggingFace, this computer) are rows that fold
 * to their name until pressed, so the card reads as three sources rather
 * than nine inputs. The fold is session state, not stored: `It never
 * remembers unless told`.
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
  const container = useStore((s) => s.container)
  const setContainer = useStore((s) => s.setContainer)
  const [driveUrl, setDriveUrl] = useState('')
  const [driveFolder, setDriveFolder] = useState('')
  /* Off, because the answer is right for the run you actually repeat: pasting
     the same folder link again to pick up the epochs uploaded since. On is for
     the one case a name cannot see — a file overwritten in place on Drive. */
  const [driveRefetch, setDriveRefetch] = useState(false)
  const [hfRepo, setHfRepo] = useState('')
  const [hfFile, setHfFile] = useState('')
  const [hfFolder, setHfFolder] = useState('')
  const [hfRefetch, setHfRefetch] = useState(false)
  const fileRef = useRef<HTMLInputElement>(null)
  const [importFiles, setImportFiles] = useState<File[]>([])
  const [importFolder, setImportFolder] = useState('')
  const [importSent, setImportSent] = useState<[number, number] | null>(null)
  const [importNote, setImportNote] = useState<string | ApiError | null>(null)
  // The whole ApiError, not `r.error`: the sentence is what the box shows, but a delete
  // that failed on a path guard answers with the route's own report behind it, and
  // widening the state is all it costs to keep that reachable.
  const [loraError, setLoraError] = useState<string | ApiError | null>(null)
  /* Which families are unfolded, keyed by name; unset means "open unless
     complete". A complete family is a list of green ticks — true, and not
     worth twenty rows of the one screen that has to be scrolled to be used.
     The head still says `complete`, so nothing is hidden that a glance was
     answering; the rows are one click away. */
  const [openFams, setOpenFams] = useState<Record<string, boolean>>({})
  /* Sections and source rows, same rule, keyed by name; unset means the
     default the section argues for above. One map for both depths because
     they are one gesture. */
  const [folds, setFolds] = useState<Record<string, boolean>>({})
  const isOpen = (k: string, dflt: boolean) => folds[k] ?? dflt
  const toggle = (k: string, dflt: boolean) =>
    setFolds((o) => ({ ...o, [k]: !(o[k] ?? dflt) }))
  /* Which multi-file rows have their file list showing. A browser cannot save
     a folder, so export on a LoRA with epochs opens onto its files. */
  const [openFiles, setOpenFiles] = useState<Record<string, boolean>>({})
  const dl = useDownload()
  // Save and every row's ✕ on one keyed flag. `dl.busy` is the uplink, which is a
  // different thing and already had its own; this is the mutations on this sheet
  // that write and then reload, and the key is which of them is in flight.
  const { busy, run } = useBusy()

  // Grouped by family, in catalogue order. Twenty-odd flat cards is a wall you
  // scroll rather than a list you read, and the group is the unit you actually
  // decide in: you want the video stack or you do not.
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

  const saveToken = () => run('token', async () => {
    const r = await setToken(token)
    setTokenNote(failed(r) ? r.error : 'Token saved.')
    setTokenValue('')
    if (!failed(r)) onReload()
  })

  const removeLora = (l: LoraEntry) => {
    // The dialog is the entire safety net: the route unlinks and there is
    // nothing behind it. So it says how much is going, and whether it can come
    // back — two different sentences, because a catalogue LoRA is a download
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
    // Keyed by root, so only the row being deleted changes; the confirm stays outside
    // `run` because nothing should be marked in flight while a dialog is still open.
    return run(`lora:${l.root}`, async () => {
      const r = await deleteLora(l.root)
      if (failed(r)) return setLoraError(r)
      setLoraError(null)
      // The whole sheet, not just this list: deleting a catalogue LoRA moves it
      // back to "missing" in the cards below.
      onReload()
    })
  }

  const pushRow = (l: LoraEntry) => {
    // Checked here as well as on the route: the route's refusal is the same
    // sentence, but a container is not the place to discover a form error.
    if (!state?.hf_token_set) {
      setLoraError('No HuggingFace token saved — a push writes to your account. '
        + 'Paste one under HuggingFace token above, with write access.')
      return
    }
    const n = l.files.length
    // A prompt rather than a field per row: the repo name is asked for once,
    // at the moment it is needed, and the default is the LoRA's own name — a
    // bare name lands under the token's account.
    const repo = prompt(
      `Push “${l.name}” to HuggingFace?\n\n`
      + `${n} file${n === 1 ? '' : 's'} (${fmtBytes(l.bytes)}) go to a private repo under `
      + 'your account — created if it does not exist, and only the files whose bytes '
      + 'differ from what is already up there are sent. Repo name, or owner/name:',
      l.name.replace(/[^A-Za-z0-9_.-]+/g, '-'),
    )
    if (repo == null || !repo.trim()) return
    setLoraError(null)
    void dl.begin(`push:${l.root}`, 'hf_push', 'Pushed.',
      () => pushLora(l.root, repo.trim()), onReload)
  }

  const doImport = () => run('import', async () => {
    setImportNote(null)
    setImportSent([0, importFiles.reduce((a, f) => a + f.size, 0)])
    const r = await uploadLoras(importFiles, importFolder.trim() || undefined,
      (sent, total) => setImportSent([sent, total]))
    setImportSent(null)
    if (failed(r)) return setImportNote(r)
    const where = importFolder.trim() ? `loras/${importFolder.trim()}/` : 'loras/'
    const got = r.files?.length ?? 0
    setImportNote(`${got} file${got === 1 ? '' : 's'} (${fmtBytes(r.bytes ?? 0)}) landed in ${where}`
      + (r.skipped?.length ? ` · skipped ${r.skipped.join(', ')}` : ''))
    setImportFiles([])
    if (fileRef.current) fileRef.current.value = ''
    onReload()
  })

  const loras = state?.loras ?? []
  const loraBytes = loras.reduce((a, l) => a + (Number(l.bytes) || 0), 0)
  const captioners = state?.caption_models ?? []
  const missing = families.flatMap((f) => f.items.filter((m) => !m.present))
  const missingGb = missing.reduce((a, m) => a + m.approx_gb, 0)
  const drive = dl.progressOf('gdrive')
  const hf = dl.progressOf('hf')
  const tokenState = tokenNote ?? (state?.hf_token_set ? 'Token saved.' : 'No token saved.')

  return (
    <div id="settings" className="scrim">
      <div className="sheet">
        <div className="sheet-head">
          <h1 className="grow">Settings</h1>
          <button className="ico" id="settings-x" type="button" onClick={onClose}>
            <IconClose />
          </button>
        </div>

        <div className="sec-group" id="runtime">
          <h2>Runtime</h2>
          <div className="card">
            <label>GPU</label>
            {container === 'one' ? (
              <div className="row" style={{ gap: 10 }}>
                <GpuSelect id="b-gpu" label="Both" side="both" spec={state?.gpus.both} />
              </div>
            ) : (
              <div className="row" style={{ gap: 10 }}>
                <GpuSelect id="g-gpu" label="Image" side="image" spec={state?.gpus.image} />
                <GpuSelect id="v-gpu" label="Video" side="video" spec={state?.gpus.video} />
              </div>
            )}
            {/* The trade is stated before it is made, in the confirm, and again
                under the control once it is — the second so the reason a picture
                is waiting is on screen when it waits. */}
            <label className="row" style={{ gap: 7, margin: '9px 0 0', color: '#ddd', fontSize: 13 }}>
              <input type="checkbox" id="one-container" style={{ width: 'auto' }}
                     checked={container === 'one'}
                     onChange={(e) => {
                       const next = e.target.checked ? 'one' : 'two'
                       if (confirm(next === 'one'
                         ? 'One container for both?\n\nOne card holds both models, so one '
                           + 'container stays warm instead of two — and a picture waits behind '
                           + 'a clip in progress. The shared container is cold, so the next run '
                           + 'pays a cold start while it loads.'
                         : 'Back to a container per family?\n\nThe next run lands on that '
                           + "family's own container, which may be cold.")) {
                         setContainer(next)
                       }
                     }} />
              One container for both
            </label>
            <p className="muted" style={{ margin: '9px 2px 0' }}>
              {container === 'one'
                ? 'On H200 both models stay loaded. On H100 or L40S they take turns through '
                  + 'memory, so switching between a picture and a clip costs a reload. '
                  + 'A picture queues behind a clip in progress.'
                : 'Changing a card costs one cold start while the model loads. Runs after it are warm.'}
            </p>
          </div>
        </div>

        <div className="sec-group" id="model-group">
          <h2>Models</h2>

          {/* The token, and only the token, folded to its state. It serves
              all three sections — gated checkpoints, private HuggingFace
              pulls, every push — so it sits above them rather than inside
              one. "Download missing" used to sit in this row, which put the
              one button that pulls the entire catalogue next to a password
              field it has nothing to do with. */}
          <div className="card" style={{ padding: '6px 16px' }}>
            <FoldRow id="tok-row" label="HuggingFace token" open={isOpen('token', false)}
                     onToggle={() => toggle('token', false)}
                     state={<span id="tok-state" className={tokenNote || state?.hf_token_set ? 'ok' : 'warn'}>
                       {tokenState}
                     </span>}>
              <div className="row">
                <input id="tok" type="password" className="grow" placeholder="hf_…"
                       autoComplete="off" value={token}
                       onChange={(e) => setTokenValue(e.target.value)} />
                {/* The note beside the row is written *by* the reply, so until the
                    reply landed there was nothing on screen at all — and the only
                    honest read of a save that shows nothing is that Save did not
                    work, which is why it got pressed again. The label carries the
                    state; the guard is `run`'s, because `disabled` is a paint and a
                    queued second click gets past a paint. */}
                <button className="s" type="button" disabled={!!busy}
                        onClick={() => void saveToken()}>
                  {busy === 'token' ? 'Saving…' : 'Save'}
                </button>
              </div>
              <p className="muted" style={{ marginTop: 8 }}>
                Needed for Krea 2 RAW and Turbo, which are gated — accept the licence at
                huggingface.co/krea/Krea-2-Raw with the same account — and for pulling
                private repos and pushing LoRAs to your own, which wants write access.
              </p>
            </FoldRow>
          </div>

          <Section id="sec-loras" name="LoRAs" open={isOpen('loras', true)}
                   onToggle={() => toggle('loras', true)}
                   summary={<span id="lora-total">
                     {loras.length ? `${loras.length} · ${fmtBytes(loraBytes)}` : 'none yet'}
                   </span>}>
            {/* Where LoRAs come from: three rows, one card, each folded to its
                name and its last word — a transfer's rate while it runs, its
                result once it has. The same card shape for all three,
                deliberately: three places weights come from, not three kinds
                of thing. */}
            <div className="card" id="lora-sources" style={{ padding: '6px 16px' }}>
              <FoldRow id="src-drive" label="Google Drive" open={isOpen('drive', false)}
                       onToggle={() => toggle('drive', false)} state={drive.message}>
                <div className="row">
                  <input id="gd-url" className="grow" placeholder="Drive link or file id"
                         autoComplete="off" spellCheck={false} value={driveUrl}
                         onChange={(e) => setDriveUrl(e.target.value)} />
                  {/* 158px is what it wants, not what it insists on. This was
                      `width:158;flex:none`, and a `.row` whose items all refuse to shrink can
                      only overflow — the link field, this one and Download ran off the right
                      edge of the sheet on a phone, and Settings has no media query of its own
                      to catch it. `minWidth` has to be spelled out too: a flex item's automatic
                      minimum is its intrinsic width, and for an `<input>` that is the default
                      ~20-character box, which is most of the overflow on its own. */}
                  <input id="gd-folder" placeholder="folder (optional)" autoComplete="off"
                         spellCheck={false} style={{ flex: '1 1 158px', minWidth: 96 }}
                         title="Group the files under loras/{name}/ — for a matched pair that belongs together. Leave blank to drop them in loose."
                         value={driveFolder} onChange={(e) => setDriveFolder(e.target.value)} />
                  {/* `.s`, matching Save in the token row. This was the only white
                      primary button on the screen, which inverted the hierarchy of
                      the whole thing: pulling one file somebody sent you was drawn
                      louder than the catalogue, which pulls 17 GB. */}
                  <button className="s" type="button" disabled={dl.busy}
                          onClick={() => {
                            if (!driveUrl.trim()) return
                            void dl.begin('gdrive', 'dl_gdrive', 'Downloaded.',
                              () => startGdrive(driveUrl.trim(), driveFolder.trim() || undefined,
                                                driveRefetch),
                              onReload)
                          }}>
                    Download
                  </button>
                </div>
                {/* Under the row rather than in it: it is a property of the pull, not a
                    third field of the link, and a folder is the only shape it changes. */}
                <label className="row" style={{ gap: 7, margin: '9px 0 0', color: '#ddd', fontSize: 13 }}>
                  <input type="checkbox" id="gd-refetch" style={{ width: 'auto' }}
                         checked={driveRefetch}
                         onChange={(e) => setDriveRefetch(e.target.checked)} />
                  Re-download files already here
                </label>
                <p className="muted" style={{ marginTop: 8 }}>
                  Lands in <code>loras/</code>, ready to name in a prompt. Only{' '}
                  <code>.safetensors</code> is kept — a folder's preview images and readme are
                  never fetched. A folder is listed before it is pulled, so a name already in{' '}
                  <code>loras/</code> costs nothing and pasting the same link again fetches only
                  what is new. Drive gives no size or checksum, so a file replaced under the name
                  it had needs the box. The link has to be shared with anyone who has it.
                </p>
                {/* No progress bar, unlike the catalogue. Drive does not say how big
                    a file is before it sends it, so a bar here could only sit at zero
                    for the length of the transfer — which is what "stuck" looks like.
                    The byte count and the rate move, and moving is the whole job. */}
                <Line p={drive} onCancel={() => void dl.cancel('gdrive', 'dl_gdrive')} />
              </FoldRow>

              <FoldRow id="src-hf" label="HuggingFace" open={isOpen('hf', false)}
                       onToggle={() => toggle('hf', false)} state={hf.message}>
                <div className="row">
                  {/* Twice the folder's share: the repo is the field, and at the
                      Drive row's split its placeholder was cut mid-word on a laptop. */}
                  <input id="hf-repo" placeholder="owner/repo, or a link to a file in one"
                         style={{ flex: '2 1 200px', minWidth: 140 }}
                         autoComplete="off" spellCheck={false} value={hfRepo}
                         onChange={(e) => setHfRepo(e.target.value)} />
                  <input id="hf-folder" placeholder="folder (optional)" autoComplete="off"
                         spellCheck={false} style={{ flex: '1 1 158px', minWidth: 96 }}
                         title="Group the files under loras/{name}/ — for a matched pair that belongs together. Leave blank to drop them in loose."
                         value={hfFolder} onChange={(e) => setHfFolder(e.target.value)} />
                  <button className="s" type="button" disabled={dl.busy}
                          onClick={() => {
                            if (!hfRepo.trim()) return
                            void dl.begin('hf', 'dl_hf', 'Downloaded.',
                              () => startHfLora(hfRepo.trim(), hfFile.trim() || undefined,
                                                hfFolder.trim() || undefined, hfRefetch),
                              onReload)
                          }}>
                    Download
                  </button>
                </div>
                <div className="row" style={{ marginTop: 8 }}>
                  {/* Its own field rather than parsed only off the link, because a
                      repo with twenty epochs is the normal case and the file you want
                      is a name you can type. Blank pulls every .safetensors the repo
                      has, up to the job's cap. */}
                  <input id="hf-file" className="grow" placeholder="file (optional) — blank takes every .safetensors in the repo"
                         autoComplete="off" spellCheck={false} value={hfFile}
                         onChange={(e) => setHfFile(e.target.value)} />
                </div>
                <label className="row" style={{ gap: 7, margin: '9px 0 0', color: '#ddd', fontSize: 13 }}>
                  <input type="checkbox" id="hf-refetch" style={{ width: 'auto' }}
                         checked={hfRefetch}
                         onChange={(e) => setHfRefetch(e.target.checked)} />
                  Re-download files already here
                </label>
                <p className="muted" style={{ marginTop: 8 }}>
                  The repo is listed first, so a name already in <code>loras/</code> costs
                  nothing and a repaste fetches only what is new. A private repo — including
                  one you pushed from here — needs the token saved above. A repo holding more
                  than twenty weights is refused with the count: name the file you want.
                </p>
                <Line p={hf} onCancel={() => void dl.cancel('hf', 'dl_hf')} />
              </FoldRow>

              <FoldRow id="src-local" label="This computer" open={isOpen('local', false)}
                       onToggle={() => toggle('local', false)}
                       state={importSent
                         ? `${fmtBytes(importSent[0])} of ${fmtBytes(importSent[1])}`
                         : typeof importNote === 'string' ? importNote : ''}>
                {/* The native picker is behind a pill: an unstyled file input is
                    the one control on this sheet that cannot lose its chrome. */}
                <input ref={fileRef} type="file" accept=".safetensors" multiple hidden
                       id="lora-files"
                       onChange={(e) => {
                         setImportNote(null)
                         setImportFiles(Array.from(e.target.files ?? []))
                       }} />
                <div className="row">
                  <button className="s" type="button" disabled={!!busy}
                          onClick={() => fileRef.current?.click()}>
                    Choose files…
                  </button>
                  <span className="grow muted" style={{ minWidth: 0, overflow: 'hidden',
                                                        textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                    {importFiles.length
                      ? `${importFiles.length} file${importFiles.length === 1 ? '' : 's'} · `
                        + `${fmtBytes(importFiles.reduce((a, f) => a + f.size, 0))} · `
                        + importFiles.map((f) => f.name).join(', ')
                      : 'nothing chosen'}
                  </span>
                  <input id="local-folder" placeholder="folder (optional)" autoComplete="off"
                         spellCheck={false} style={{ flex: '1 1 158px', minWidth: 96 }}
                         title="Group the files under loras/{name}/ — for a matched pair that belongs together. Leave blank to drop them in loose."
                         value={importFolder} onChange={(e) => setImportFolder(e.target.value)} />
                  <button className="s" type="button" disabled={!!busy || !importFiles.length}
                          onClick={() => void doImport()}>
                    {busy === 'import' ? 'Importing…' : 'Import'}
                  </button>
                </div>
                {importSent && (
                  <div className="fam-prog" style={{ marginTop: 10 }}>
                    <div className="bar" style={{ marginTop: 0 }}>
                      <i style={{ width: `${importSent[1] ? importSent[0] / importSent[1] * 100 : 0}%` }} />
                    </div>
                    <p className="muted" style={{ margin: '7px 0 0' }}>
                      Uploading · {fmtBytes(importSent[0])} of {fmtBytes(importSent[1])}
                    </p>
                  </div>
                )}
                {typeof importNote === 'string'
                  ? <p className="ok" style={{ marginTop: 8 }}>{importNote}</p>
                  : <ErrorNote err={importNote} style={{ marginTop: 10 }} />}
                <p className="muted" style={{ marginTop: 8 }}>
                  Only <code>.safetensors</code>. A file with a name already in{' '}
                  <code>loras/</code> replaces it.
                </p>
              </FoldRow>
            </div>

            <div className="card">
              <ErrorNote err={loraError} style={{ marginBottom: 10 }} />
              <div id="lora-list">
                {loras.length ? loras.map((l) => {
                  const first = l.files[0]
                  const one = l.files.length === 1 && !!first
                  const pushing = dl.progressOf(`push:${l.root}`)
                  return (
                    <div key={l.root}>
                      <div className="lora-row">
                        <div className="grow" style={{ minWidth: 0 }}>
                          <b>{l.name}</b>
                          {l.trigger_word ? <> <code>{l.trigger_word}</code></> : null}
                          <div className="muted">
                            {l.files.length} file{one ? '' : 's'} · {fmtBytes(l.bytes)}
                            {l.catalogue ? ` · ${l.catalogue}` : ''}
                          </div>
                        </div>
                        {/* Export: a plain link when there is one file, because a
                            download is what a link already is; a fold onto the files
                            when there are epochs, because a browser cannot save a
                            folder and picking one is a choice worth seeing. */}
                        {one ? (
                          <a className="lora-act lora-export" href={loraFileUrl(first.path ?? '')}
                             download={first.name} title="Download to this computer">
                            <IconDownload />
                          </a>
                        ) : (
                          <button className="lora-act lora-export" type="button"
                                  aria-expanded={!!openFiles[l.root]}
                                  title={`Download to this computer — ${l.files.length} files`}
                                  onClick={() => setOpenFiles((o) => ({ ...o, [l.root]: !o[l.root] }))}>
                            <IconDownload />
                          </button>
                        )}
                        <button className="lora-act lora-push" type="button" disabled={dl.busy}
                                title="Push to HuggingFace (private)"
                                onClick={() => pushRow(l)}>
                          <IconUpload />
                        </button>
                        {/* Unlinking the files and reloading the sheet leaves the row
                            sitting there, still with a ✕ on it, for as long as both take
                            — which reads as a delete that did not take, and the second
                            press opens the confirm again for a LoRA already on its way
                            out. The glyph carries it because `.lora-x` is a fixed
                            square: there is no room in it for a word. */}
                        <button className="lora-x" type="button" disabled={!!busy}
                                title={busy === `lora:${l.root}` ? 'Deleting…' : 'Delete'}
                                onClick={() => void removeLora(l)}>
                          {busy === `lora:${l.root}` ? '…' : <IconClose />}
                        </button>
                      </div>
                      {!one && openFiles[l.root] && (
                        <div className="lora-files">
                          {l.files.map((f) => (
                            <div key={f.path ?? f.name}>
                              <code className="grow">{f.name}</code>
                              <a className="lora-act" href={loraFileUrl(f.path ?? '')}
                                 download={f.name} title={`Download ${f.name}`}>
                                <IconDownload />
                              </a>
                            </div>
                          ))}
                        </div>
                      )}
                      <Line p={pushing}
                            onCancel={() => void dl.cancel(`push:${l.root}`, 'hf_push')} />
                    </div>
                  )
                }) : (
                  // The three ways in are the card above this one, so the empty
                  // state points at them rather than saying "nothing here".
                  <p className="muted" style={{ margin: '2px 0 0' }}>
                    Nothing in <code>loras/</code> yet — train one, or bring one in above.
                  </p>
                )}
              </div>
            </div>
          </Section>

          {/* Captioners are menu rows, not catalogue entries: the weights pull
              into the HF cache on first use rather than downloading here, so the
              card offers add-by-repo instead of a Download button. */}
          <Section id="sec-captions" name="Caption models" open={isOpen('captions', false)}
                   onToggle={() => toggle('captions', false)}
                   summary={`${captioners.length} in the menu`}>
            <CaptionModels state={state} onReload={onReload} />
          </Section>

          <Section id="sec-checkpoints" name="Checkpoints"
                   open={isOpen('checkpoints', missing.length > 0)}
                   onToggle={() => toggle('checkpoints', missing.length > 0)}
                   summary={missing.length
                     ? `${missing.length} missing · ${missingGb.toFixed(1)} GB`
                     : 'complete'}>
            <div id="models">
              {families.map((f) => {
                const left = f.items.filter((m) => !m.present)
                const size = left.reduce((a, m) => a + m.approx_gb, 0)
                const id = `fam:${f.name}`
                const open = openFams[f.name] ?? left.length > 0
                return (
                  <div className="fam" key={f.name}>
                    <div className="fam-head">
                      {/* The name is the toggle; the caret is what says so. A
                          separate disclosure control beside the name would be two
                          marks for one act. `.fam-dl` keeps its own button because
                          a toggle and a 17 GB download must not share a target. */}
                      <button className="t fam-toggle" type="button" aria-expanded={open}
                              onClick={() => setOpenFams((o) => ({ ...o, [f.name]: !open }))}>
                        <span className={`caret${open ? ' open' : ''}`} aria-hidden="true" />
                        <b>{f.name}</b>
                      </button>
                      <span className="muted">
                        {left.length ? `${left.length} missing · ${size.toFixed(1)} GB` : 'complete'}
                      </span>
                      {/* One short shows no button at all, because that is what its
                          own Download already is. */}
                      {/* A family is the unit you decide in — you want the stack or
                          you do not — so this is the press that matters, and the
                          per-file buttons below are the escape hatch. A quiet pill
                          now rather than the screen's one `.b`: the mockup's call,
                          and the white fill was drawing a purchase decision louder
                          than the canvas draws Generate. */}
                      {left.length > 1 && (
                        <button className="s fam-dl" type="button" disabled={dl.busy}
                                onClick={() => void dl.begin(id, id, `${f.name} downloaded.`,
                                  () => downloadFamily(f.name, token), onReload)}>
                          Download all {left.length}
                        </button>
                      )}
                    </div>
                    <Line p={dl.progressOf(id)} bar onCancel={() => void dl.cancel(id, id)} />

                    {/* One card per family, rows inside — the LoRA list's own
                        shape, and the caption menu's. Twenty sibling cards made
                        the catalogue a wall of frames where the reading unit is
                        the family; a hairline per row is the same information at
                        a third of the chrome. */}
                    {open && (
                      <div className="card">
                        {f.items.map((m) => {
                          const p = dl.progressOf(m.key)
                          return (
                            <div className="mrow" key={m.key}>
                              <div className="grow" style={{ minWidth: 0 }}>
                                <b>{m.label}</b> <span className="muted">{m.note}</span>
                                {m.gated && <span className="warn" style={{ fontSize: 12 }}> · gated</span>}
                                <div className="muted" style={{ marginTop: 3 }}><code>{m.repo_id}</code></div>
                              </div>
                              <div style={{ textAlign: 'right' }}>
                                {m.present ? (
                                  <span className="ok">✓ {m.size_gb} GB</span>
                                ) : (
                                  <span className="row" style={{ gap: 12, justifyContent: 'flex-end' }}>
                                    <span className="muted" style={{ whiteSpace: 'nowrap' }}>
                                      {m.approx_gb} GB
                                    </span>
                                    {/* The size is the label, so the button can be a
                                        mark — a control whose value is beside it, in
                                        the row you are already reading. Cancel takes
                                        the slot while the pull runs: two controls in
                                        one place would be an invitation to press the
                                        dead one. */}
                                    {p.running ? (
                                      <button className="s" type="button"
                                              onClick={() => void dl.cancel(m.key, `dl_${m.key}`)}>
                                        Cancel
                                      </button>
                                    ) : (
                                      <button className="icx" type="button" disabled={dl.busy}
                                              title={`Download ${m.approx_gb} GB`}
                                              onClick={() => void dl.begin(m.key, `dl_${m.key}`, 'Done',
                                                () => startDownload(m.key), onReload)}>
                                        <IconDownload />
                                      </button>
                                    )}
                                  </span>
                                )}
                                {p.tone === 'err' ? (
                                  // Left-aligned and capped by hand: this column is
                                  // `text-align:right` and sized by its own content, so an
                                  // err-box dropped into it would right-align a traceback and
                                  // stretch the card until the label beside it wrapped a word
                                  // per line.
                                  <ErrorNote err={{ error: p.message, detail: p.detail }}
                                             style={{ textAlign: 'left', maxWidth: 280, margin: '8px 0 0' }} />
                                ) : (
                                  <div className={`muted dl-state${p.tone === 'ok' ? ' ok' : ''}`}>
                                    {p.message}
                                  </div>
                                )}
                              </div>
                            </div>
                          )
                        })}
                      </div>
                    )}
                  </div>
                )
              })}
            </div>
          </Section>
        </div>
      </div>
    </div>
  )
}

/**
 * A section of the sheet: a name that folds, and a summary that stays.
 *
 * The summary is the part that must survive the fold — the LoRA total, the
 * missing gigabytes, the size of the caption menu — so nothing a glance was
 * answering goes behind the click. Same handle as a family's, because a
 * section head and a family head are the same gesture at two depths.
 */
function Section({ id, name, summary, open, onToggle, children }: {
  id: string
  name: string
  summary: ReactNode
  open: boolean
  onToggle: () => void
  children: ReactNode
}) {
  return (
    <div className={`sec${open ? '' : ' shut'}`} id={id}>
      <div className="sec-head">
        <button className="t fam-toggle" type="button" aria-expanded={open} onClick={onToggle}>
          <span className={`caret${open ? ' open' : ''}`} aria-hidden="true" />
          <b>{name}</b>
        </button>
        <span className="muted">{summary}</span>
      </div>
      {open && children}
    </div>
  )
}

/**
 * A row inside a card that is a form once pressed.
 *
 * `state` is its last word while folded — a transfer's rate, a result, the
 * token's saved-or-not — because a source that is quietly moving 2 GB must
 * not look idle just because its form is closed.
 */
function FoldRow({ id, label, state, open, onToggle, children }: {
  id: string
  label: string
  state?: ReactNode
  open: boolean
  onToggle: () => void
  children: ReactNode
}) {
  return (
    <>
      <button className="fold-row" id={id} type="button" aria-expanded={open} onClick={onToggle}>
        <span className={`caret${open ? ' open' : ''}`} aria-hidden="true" />
        {label}
        <span className="muted">{state}</span>
      </button>
      {open && <div className="fold-body">{children}</div>}
    </>
  )
}

/**
 * The captioner menu, editable.
 *
 * Any vision-language model on HuggingFace, by repo id. The add is validated
 * server-side against the repo's own config.json — a typo, a gated repo
 * without a token, or a text-only model is a named error here, in
 * milliseconds, rather than a cold GPU start that dies mid-pull. Every row has
 * a ✕: a custom is deleted from the config Dict, and a built-in — baked into
 * the image, so not deletable — is hidden through the same Dict, which is what
 * lets the hide survive a redeploy. One gesture either way: the entry leaves
 * the menu, and adding the repo again brings it back.
 */
function CaptionModels({ state, onReload }: {
  state: AppState | null
  onReload: () => void
}) {
  const [repo, setRepo] = useState('')
  const [label, setLabel] = useState('')
  const [note, setNote] = useState<{ text: string; err?: boolean } | null>(null)
  // Both paths on one flag. Add had a hand-rolled `busy` boolean and Remove had nothing,
  // so the same card said "Checking…" for one button and sat silent for the other — and
  // two mechanisms for one behaviour is how they drift apart.
  const { busy, run } = useBusy()
  const models = state?.caption_models ?? []

  const add = () => {
    // Only the empty-field check is left here: `run` refuses while anything is in
    // flight, which is the half the old `|| busy` was doing.
    if (!repo.trim()) return
    return run('add', async () => {
      setNote(null)
      const r = await addCaptionModel(repo.trim(), label.trim())
      if (failed(r)) return setNote({ text: r.error, err: true })
      setRepo('')
      setLabel('')
      setNote({ text: 'Added. The weights pull into the cache on its first run.' })
      onReload()
    })
  }

  const remove = (key: string, name: string) => {
    if (!confirm(`Remove “${name}” from the captioner menu?\n\n`
      + 'Only the menu entry goes — weights already in the cache stay cached. '
      + 'Add the repo again to bring it back.')) return
    return run(`rm:${key}`, async () => {
      const r = await deleteCaptionModel(key)
      if (failed(r)) return setNote({ text: r.error, err: true })
      onReload()
    })
  }

  return (
    <div className="card" id="caption-models">
      {models.map((m) => (
        <div className="lora-row" key={m.key}>
          <div className="grow" style={{ minWidth: 0 }}>
            <b>{m.label}</b> <span className="muted">{m.note}</span>
            {m.repo && <div className="muted" style={{ marginTop: 3 }}><code>{m.repo}</code></div>}
          </div>
          {/* Same treatment as the LoRA rows above: the row survives the round trip and
              the reload, so without this the press had no answer until the list redrew.
              Built-ins carry it too — the route hides them rather than deleting, and
              refuses the last row, because a menu with no captioner in it is a
              captioning run with nothing to offer. */}
          <button className="lora-x" type="button" disabled={!!busy}
                  title={busy === `rm:${m.key}` ? 'Removing…' : 'Remove from the menu'}
                  onClick={() => void remove(m.key, m.label)}>
            {busy === `rm:${m.key}` ? '…' : <IconClose />}
          </button>
        </div>
      ))}
      <div className="row" style={{ marginTop: 10 }}>
        <input id="cm-repo" className="grow" placeholder="owner/repo — any vision LM on HuggingFace"
               autoComplete="off" spellCheck={false} value={repo}
               onChange={(e) => setRepo(e.target.value)}
               onKeyDown={(e) => { if (e.key === 'Enter') void add() }} />
        {/* Same basis and the same explicit minimum as the Drive folder field, for the
            same reason: this row is repo + label + Add, and a fixed-width middle is what
            pushed Add off the edge on a narrow screen. */}
        <input id="cm-label" placeholder="label (optional)" autoComplete="off"
               spellCheck={false} style={{ flex: '1 1 158px', minWidth: 96 }} value={label}
               onChange={(e) => setLabel(e.target.value)} />
        <button className="s" type="button" disabled={!!busy || !repo.trim()}
                onClick={() => void add()}>
          {busy === 'add' ? 'Checking…' : 'Add'}
        </button>
      </div>
      {note && (
        <p className={note.err ? 'err' : 'ok'} style={{ marginTop: 8 }}>{note.text}</p>
      )}
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
/** What a card costs beyond the cold start, said before it is chosen. Only the
 *  48 GB card has something to say: it is the one where a graph that fits
 *  elsewhere may not, and the one where video pays in speed. The numbers are the
 *  backend's (`IMAGE_GPUS` in app.py); the sentence lives here because the
 *  confirm is where it is read. */
function switchNote(card: string, side: 'image' | 'video' | 'both'): string {
  if (card !== 'L40S') return ''
  const image = 'A plain picture fits in its 48 GB; a regional render with reference '
    + 'photos may not. '
  const video = 'Video runs with its weights paged through 48 GB, several times slower '
    + 'than an H100. '
  return side === 'image' ? image : side === 'video' ? video : image + video
}
function GpuSelect({ id, label, side, spec }: {
  id: string
  label: string
  side: 'image' | 'video' | 'both'
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
                    || confirm(`Switch to ${next}?\n\n${switchNote(next, side)}`
                      + 'This card has no warm container, so the next run pays a cold '
                      + 'start while the model loads. Runs after it are warm.')) {
                  setGpu({ [side]: next })
                }
              }}>
        {(spec?.options ?? []).map((o) => <option key={o}>{o}</option>)}
      </select>
    </div>
  )
}

function Line({ p, bar, onCancel }: {
  p: { percent: number; message: string; tone: string; running: boolean; detail?: string }
  bar?: boolean
  onCancel: () => void
}) {
  if (!p.message) return null
  // A failure gets the box and its disclosure, not a red line of text. A refused start
  // is the one message here with a server behind it — "could not reach the server" with
  // the browser's own reason underneath — and a bare `<p className="err">` had nowhere
  // to put the reason, so it was dropped. Bar and Cancel are both gone by then anyway:
  // `running` is false on every path that sets this tone.
  if (p.tone === 'err') {
    return (
      <div className="fam-prog">
        <ErrorNote err={{ error: p.message, detail: p.detail }} style={{ marginBottom: 0 }} />
      </div>
    )
  }
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
