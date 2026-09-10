/**
 * The Playground — the backend's own graph on screen, rewired by hand.
 *
 * A room entered by a door, on the Sheet precedent: experimenting on the
 * engine is neither generating nor training. The veto list governs the
 * product canvas; this is the lab, and even here nothing wears a devtools
 * costume — cards in the app's own tokens, quiet gestures, prose errors.
 *
 * It opens on the app's own live graph, built from the console's state by
 * the same builders a render uses, so nothing is ever blank and the diff
 * between the shipped graph and yours *is* the experiment. Runs land in the
 * shared gallery like any other render, with the graph embedded in the file
 * the way ComfyUI does it — the PNG is the workflow, and dropping one back
 * here reopens it.
 */
import { useCallback, useContext, useEffect, useRef, useState } from 'react'

import { failed, type ApiError } from '../api/client'
import {
  deletePack, deleteWorkflow, fileUrl, getWorkflow, listWorkflows,
  playgroundPacks, playgroundSeed, saveWorkflow,
} from '../api/routes'
import type { PackRow, WorkflowExpose, WorkflowRow } from '../api/types'
import { imageBody } from '../canvas/useGenerate'
import { useStore, type Kind, type PgGraph } from '../store'
import { ErrorNote } from '../ui/ErrorNote'
import { Menu, type MenuItem } from '../ui/Menu'
import { Popover, usePopover } from '../ui/Popover'
import { videoBody } from '../video/useVideo'
import { CatContext, defaultInputs, diffExposes, useCatalogue } from './catalogue'
import { Editor } from './Editor'
import { graphFromPng } from './png'
import { usePlayground } from './usePlayground'
import './playground.css'

export function Playground() {
  const s = useStore()
  const { cat, missing, error: catErr, reload } = useCatalogue()
  const { run, fire, cancel, restart, harvest, install, clearError } =
    usePlayground(useCallback((doing: string | null) => {
      if (doing === 'catalogue' || doing === 'pack') reload()
    }, [reload]))
  const [arrange, setArrange] = useState(1)
  const [note, setNote] = useState<string | ApiError | null>(null)
  /** Exposes loaded with a workflow, carried through a re-save — a loaded
   *  graph has no seed to diff against, and saving must not silently strip
   *  the controls the console already grew for it. */
  const kept = useRef<WorkflowExpose[]>([])

  const seedNow = useCallback(async (host?: Kind) => {
    const st = useStore.getState()
    const h = host ?? st.kind
    const body = h === 'image' ? imageBody(st) : videoBody(st)
    const r = await playgroundSeed({ ...body, kind: h })
    if (failed(r)) { setNote(r); return }
    const graph = r.graph as PgGraph
    kept.current = []
    st.setPg({ graph, seed: graph, host: h, name: '', dirty: false,
               sel: null, attachments: {} })
    setNote(r.dropped?.length
      ? { error: 'Reference pictures do not cross into a seed — attach '
                 + 'files on the nodes that read them instead.' }
      : null)
    setArrange((a) => a + 1)
  }, [])

  // The room opens on the app's own graph — nothing is ever blank. Once:
  // after that the graph is yours, and coming back into the room must not
  // overwrite it (keep.ts carries it across reloads for the same reason).
  const seeded = useRef(false)
  useEffect(() => {
    if (!seeded.current && !s.pg.graph) {
      seeded.current = true
      void seedNow()
    }
  }, [s.pg.graph, seedNow])

  const loadGraph = useCallback((graph: PgGraph, name: string,
                                 exposes: WorkflowExpose[]) => {
    kept.current = exposes
    useStore.getState().setPg({ graph, seed: null, name, dirty: false,
                                sel: null })
    setNote(null)
    setArrange((a) => a + 1)
  }, [])

  const doSave = useCallback(async (name: string) => {
    const st = useStore.getState()
    if (!st.pg.graph || !name.trim()) return
    const exposes = st.pg.seed
      ? diffExposes(st.pg.graph, st.pg.seed, cat)
      : kept.current
    const r = await saveWorkflow(name.trim(), st.pg.graph, {
      seed_of: st.pg.host, exposes, created: Date.now() / 1000,
    })
    if (failed(r)) { setNote(r); return }
    kept.current = exposes
    st.setPg({ name: name.trim(), dirty: false })
  }, [cat])

  // A dropped PNG is a workflow arriving — ComfyUI's own gesture, honoured
  // here byte for byte.
  const onDrop = useCallback(async (e: React.DragEvent) => {
    e.preventDefault()
    const f = Array.from(e.dataTransfer.files)
      .find((x) => x.name.toLowerCase().endsWith('.png'))
    if (!f) return
    try {
      loadGraph(await graphFromPng(f), '', [])
    } catch (err) {
      setNote({ error: err instanceof Error ? err.message : String(err) })
    }
  }, [loadGraph])

  const busyWord = run.doing === 'restart' ? 'Restarting the engine'
    : run.doing === 'catalogue' ? 'Reading the node catalogue'
    : run.doing === 'pack' ? 'Installing the pack'
    : run.phase

  return (
    <CatContext.Provider value={cat}>
      <div className="view playground" id="v-playground"
           onDragOver={(e) => e.preventDefault()} onDrop={onDrop}>
        <div className="pgbar">
          <EngineButton onPick={(h) => void seedNow(h)} />
          <NameField onSave={doSave} />
          <LoadButton onLoad={loadGraph} onNote={setNote} />
          <AddButton />
          <button className="opt" id="pg-tidy" type="button"
                  title="Recompute the layout"
                  onClick={() => setArrange((a) => a + 1)}>
            Tidy
          </button>
          <button className="opt" id="pg-reseed" type="button"
                  title="Replace the graph with the app's own, built from the console"
                  onClick={() => void seedNow(s.pg.host)}>
            Reset to the app's graph
          </button>
          <span className="grow" />
          <PacksButton onInstall={(u, r) => void install(u, r)}
                       onRestart={() => void restart()}
                       onHarvest={() => void harvest()} />
          {run.running ? (
            <button className="opt danger" id="pg-stop" type="button"
                    onClick={() => void cancel()}>
              Stop
            </button>
          ) : (
            <button className="go" id="pg-run" type="button"
                    disabled={!s.pg.graph}
                    onClick={() => { clearError(); void fire() }}>
              Run
            </button>
          )}
        </div>

        <div className="pgstage">
          {s.pg.graph
            ? <Editor arrange={arrange} errorNode={run.errorNode} />
            : <div className="pgempty">Reading the console…</div>}
          {missing && (
            <div className="pgnote">
              <span>
                No node catalogue yet — it is read once from ComfyUI on a CPU
                container, and the editor names every node from it.
              </span>
              <button className="opt" type="button"
                      onClick={() => void harvest()}>
                Read it now
              </button>
            </div>
          )}
          {run.running && (
            <div className="pgprog" role="status">
              <span className="pgphase">{busyWord}</span>
              {run.percent > 0 && (
                <span className="pgpct">{Math.round(run.percent)}%</span>
              )}
              <div className="pgtrack">
                <div className="pgfill"
                     style={{ width: `${Math.max(2, run.percent)}%` }} />
              </div>
            </div>
          )}
          {(run.error || note || catErr) && !run.running && (
            <div className="pgnote">
              <ErrorNote err={run.error ?? note ?? catErr!} />
              <button className="opt" type="button"
                      onClick={() => { clearError(); setNote(null) }}>
                Dismiss
              </button>
            </div>
          )}
          {run.note && !run.running && (
            <div className="pgnote quiet"><span>{run.note}</span></div>
          )}
        </div>

        {run.jobId && run.files.length > 0 && (
          <div className="pgresults">
            {run.files.map((f) => f.toLowerCase().endsWith('.png')
              || f.toLowerCase().endsWith('.jpg') ? (
                <img key={f} src={fileUrl(f)} alt={f} />
              ) : (
                <video key={f} src={fileUrl(f)} controls loop />
              ))}
          </div>
        )}
      </div>
    </CatContext.Provider>
  )
}

/** Which engine runs it — the same two warm containers the console uses. */
function EngineButton({ onPick }: { onPick: (h: Kind) => void }) {
  const host = useStore((st) => st.pg.host)
  const pop = usePopover()
  const items: MenuItem[] = [
    { label: 'Krea 2 engine', on: host === 'image',
      run: () => onPick('image') },
    { label: 'MiniMax-H3 engine', on: host === 'video',
      run: () => onPick('video') },
  ]
  return (
    <>
      <button className={`opt${pop.open ? ' on' : ''}`} id="pg-engine"
              type="button" title="Which warm engine runs the graph"
              onClick={pop.toggle}>
        {host === 'image' ? 'Krea 2' : 'MiniMax-H3'}
      </button>
      {pop.open && <Menu anchor={pop.anchor} items={items}
                         onClose={pop.close} />}
    </>
  )
}

/** The workflow's name, edited in place; Save appears once there is a graph.
 *  The dirty dot rides the field, not a modal — nothing asks for
 *  confirmation, and what replaces it is that Save is always one press. */
function NameField({ onSave }: { onSave: (name: string) => Promise<void> }) {
  const pg = useStore((st) => st.pg)
  const [draft, setDraft] = useState<string | null>(null)
  const value = draft ?? pg.name
  return (
    <span className="pgname-box">
      <input autoComplete="off" className="pgwf" type="text" placeholder="untitled"
             value={value} id="pg-name"
             onChange={(e) => setDraft(e.target.value)}
             onKeyDown={(e) => {
               if (e.key === 'Enter' && value.trim()) {
                 void onSave(value).then(() => setDraft(null))
               }
             }} />
      {pg.dirty && <span className="pgdot" title="Unsaved changes" />}
      <button className="opt" id="pg-save" type="button"
              disabled={!value.trim() || !pg.graph}
              onClick={() => void onSave(value).then(() => setDraft(null))}>
        Save
      </button>
    </span>
  )
}

function LoadButton({ onLoad, onNote }: {
  onLoad: (g: PgGraph, name: string, exposes: WorkflowExpose[]) => void
  onNote: (e: ApiError) => void
}) {
  const pop = usePopover()
  const [rows, setRows] = useState<WorkflowRow[] | null>(null)
  const open = async (e: React.MouseEvent<HTMLElement>) => {
    pop.toggle(e)
    const r = await listWorkflows()
    setRows(failed(r) ? [] : r.workflows)
  }
  const current = useStore((st) => st.pg.name)
  const items: MenuItem[] = (rows ?? []).map((w) => ({
    label: w.name,
    on: w.name === current,
    run: async () => {
      const r = await getWorkflow(w.name)
      if (failed(r)) { onNote(r); return }
      onLoad(r.graph as PgGraph, r.name, r.meta.exposes ?? [])
    },
  }))
  if (current) {
    items.push({ sep: true }, {
      label: `Delete ${current}`, danger: true,
      run: async () => {
        const r = await deleteWorkflow(current)
        if (failed(r)) { onNote(r); return }
        useStore.getState().setPg({ name: '', dirty: true })
      },
    })
  }
  return (
    <>
      <button className={`opt${pop.open ? ' on' : ''}`} id="pg-load"
              type="button" title="Saved workflows, from workflows/ on the volume"
              onClick={(e) => void open(e)}>
        Load
      </button>
      {pop.open && (
        <Menu anchor={pop.anchor} onClose={pop.close}
              items={items.length ? items
                : [{ label: rows ? 'Nothing saved yet' : 'Loading…',
                     run: () => {} }]} />
      )}
    </>
  )
}

/** Add a node — search the catalogue, get the node with every widget input
 *  written out at the spec's default (the backend's own rule: an omitted
 *  optional input is decided by a signature nobody can see). */
function AddButton() {
  const cat = useContext(CatContext)
  const pop = usePopover()
  const [q, setQ] = useState('')
  const needle = q.trim().toLowerCase()
  const hits = cat
    ? Object.keys(cat)
        .filter((k) => !needle
          || k.toLowerCase().includes(needle)
          || String(cat[k]?.display_name ?? '').toLowerCase().includes(needle))
        .slice(0, 24)
    : []
  const add = (classType: string) => {
    const st = useStore.getState()
    const g = st.pg.graph ?? {}
    const numeric = Object.keys(g).map((k) => parseInt(k, 10))
      .filter(Number.isFinite)
    let key = String((numeric.length ? Math.max(...numeric) : 0) + 1)
    while (key in g) key = `${key}_`
    st.setPg({
      graph: { ...g, [key]: { class_type: classType,
                              inputs: defaultInputs(cat?.[classType]) } },
      dirty: true, sel: key,
    })
  }
  return (
    <>
      <button className={`opt${pop.open ? ' on' : ''}`} id="pg-add"
              type="button" title="Add a node" onClick={pop.toggle}>
        + Node
      </button>
      {pop.open && (
        <Popover anchor={pop.anchor} className="form pgadd"
                 onClose={() => { pop.close(); setQ('') }}>
          <input autoComplete="off" type="text" placeholder="Search nodes…" value={q} autoFocus
                 onChange={(e) => setQ(e.target.value)} />
          <div className="pgadd-list">
            {!cat && (
              <span className="pgadd-none">
                The catalogue has not been read yet.
              </span>
            )}
            {cat && !hits.length && (
              <span className="pgadd-none">Nothing matches.</span>
            )}
            {hits.map((k) => (
              <button key={k} type="button" onClick={() => {
                add(k)
                pop.close()
                setQ('')
              }}>
                {String(cat?.[k]?.display_name ?? k)}
                <span className="hint">{String(cat?.[k]?.category ?? '')}</span>
              </button>
            ))}
          </div>
        </Popover>
      )}
    </>
  )
}

function PacksButton({ onInstall, onRestart, onHarvest }: {
  onInstall: (url: string, ref?: string) => void
  onRestart: () => void
  onHarvest: () => void
}) {
  const pop = usePopover()
  const [rows, setRows] = useState<PackRow[] | null>(null)
  const [url, setUrl] = useState('')
  const open = async (e: React.MouseEvent<HTMLElement>) => {
    pop.toggle(e)
    const r = await playgroundPacks()
    setRows(failed(r) ? [] : r.packs)
  }
  return (
    <>
      <button className={`opt${pop.open ? ' on' : ''}`} id="pg-packs"
              type="button" title="Node packs installed from git"
              onClick={(e) => void open(e)}>
        Packs
      </button>
      {pop.open && (
        <Popover anchor={pop.anchor} className="form pgpacks"
                 onClose={pop.close}>
          {(rows ?? []).map((p) => (
            <div className="pgpack" key={p.name}>
              <span className="pgpack-name" title={p.url ?? ''}>{p.name}</span>
              <span className="hint">{p.sha}</span>
              <button className="opt danger" type="button" title="Delete this pack"
                      onClick={async () => {
                        await deletePack(p.name)
                        const r = await playgroundPacks()
                        setRows(failed(r) ? [] : r.packs)
                      }}>
                ×
              </button>
            </div>
          ))}
          {rows && !rows.length && (
            <span className="pgadd-none">No packs installed.</span>
          )}
          <div className="pgpack-add">
            <input autoComplete="off" type="text" placeholder="https://github.com/owner/repo"
                   value={url} onChange={(e) => setUrl(e.target.value)} />
            <button className="opt" type="button" disabled={!url.trim()}
                    onClick={() => {
                      pop.close()
                      onInstall(url.trim())
                      setUrl('')
                    }}>
              Install
            </button>
          </div>
          <hr />
          <button className="opt" type="button"
                  title="Kill the warm ComfyUI and start a fresh one — loads new packs, clears anything a pack wedged. The next run reloads its checkpoint."
                  onClick={() => { pop.close(); onRestart() }}>
            Restart the engine
          </button>
          <button className="opt" type="button"
                  title="Re-read /object_info on a CPU container"
                  onClick={() => { pop.close(); onHarvest() }}>
            Re-read the catalogue
          </button>
        </Popover>
      )}
    </>
  )
}
