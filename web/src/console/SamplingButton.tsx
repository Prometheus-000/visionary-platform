import { useEffect, useState } from 'react'

import { failed } from '../api/client'
import { listWorkflows } from '../api/routes'
import type { WorkflowExpose, WorkflowRow as Workflow } from '../api/types'
import { Popover, usePopover } from '../ui/Popover'
import { NumInput } from '../ui/NumInput'
import { useStore, videoModel } from '../store'
import { resolveVid, vidEdited } from './resolve'

/**
 * Everything you touch rarely, behind one button.
 *
 * "Advanced" was a drawer, which is a name for where something is rather than what
 * it does — and behind it sat five controls that are not advanced, they are just
 * rarely changed. A drawer also charged the console a whole row the moment you
 * opened it to read one number.
 *
 * **The button is named for the model**, because the model is the rarely-touched
 * choice that decides what every frequently-touched one *means*: which sizes exist,
 * whether there is a negative prompt to write, whether LoRAs load at all. Naming it
 * `8 steps · CFG 1.0` named it after two values nobody chose — those are the
 * checkpoint's defaults. The checkpoint is the choice; the numbers are its
 * consequences.
 *
 * A form rather than a palette: a sampler name and a step count are not things a
 * picture of them could teach, which is the line the shot tiles are on the other
 * side of.
 */
function Row({ label, hint, children }: {
  label: string; hint?: string; children: React.ReactNode
}) {
  return (
    <label className="frow">
      <span>{label}</span>
      {children}
      {hint && <i>{hint}</i>}
    </label>
  )
}

// The carve-out to the copy-is-a-last-resort rule, and the same one the
// dimension boxes get: an empty field cannot show what emptiness does. There is
// no lock glyph, no dice and no chip — the number is visible in its own field
// and one gesture from gone, which is what makes saying so enough.
//
// **It said the seed pinned itself here, and that stopped being true.** The
// pin fired only once the prompt had been read into a document; there are no
// documents now, so the field is always what you left it. Copy describing a
// behaviour the code no longer has is worse than no copy, and it is the kind
// that survives a deletion because nobody greps the strings.
const SEED_HINT = 'Blank draws a new one. A seed worth keeping is on the render '
  + 'that used it, and Reuse puts it back.'
const SHIFT_HINT = 'Bends the noise schedule — higher spends more steps on composition and motion.'

/**
 * **Switching the model is how you cross between the two consoles.**
 *
 * Image and video are sibling disciplines with their own console and their own
 * canvas — a photographer is not a filmmaker with the duration turned down — so
 * there is a door between them, and this is it. One list, both families, and
 * picking from the other one takes the whole surface with it.
 *
 * It belongs here by this button's own definition: the model is the
 * rarely-touched choice that decides what every frequently-touched one *means*,
 * and Krea 2 against H3 is that difference at its largest — which sizes exist,
 * whether there is a negative prompt to write, whether there is a prompt box at
 * all. The list used to be forked by the console you were already in, which
 * made it a picker you could only reach a family through after arriving there.
 *
 * Duration is *not* the door and must not become one again. Zero is a length
 * both consoles can answer, so going still never changes the engine; adding
 * time does, because Krea 2 cannot answer it. See `Duration`.
 */
type Engine = { key: string; label: string; kind: 'image' | 'video'; ready: boolean }

function useEngines(): Engine[] {
  const s = useStore()
  // Turbo first, against the catalogue's order, for the reason `ImageSampling`
  // already records: the catalogue is ordered by what trains and this picker is
  // read by what generates.
  const img: Engine[] = ['turbo', 'raw']
    .map((k) => s.state?.models.find((m) => m.key === k))
    .filter((m): m is NonNullable<typeof m> => !!m)
    .map((m) => ({ key: m.key, label: m.label, kind: 'image', ready: !!m.present }))
  const vid: Engine[] = (s.state?.video_models ?? [])
    .map((v) => ({ key: v.key, label: v.label, kind: 'video', ready: !!v.ready }))
  return [...img, ...vid]
}

function EngineRow({ current, onCross }: { current: string; onCross: () => void }) {
  const s = useStore()
  const list = useEngines()
  // Keyed `kind:key` rather than by key alone. The two families do not collide
  // today, and a picker whose correctness depends on that is one an added
  // checkpoint breaks silently — it would switch the wrong console and look
  // like the model not taking.
  const value = `${s.kind}:${current}`
  return (
    <Row label="Model">
      <select value={value} onChange={(e) => {
        const hit = list.find((x) => `${x.kind}:${x.key}` === e.target.value)
        if (!hit) return
        if (hit.kind === 'image') s.setImg({ model: hit.key })
        else s.setVid({ model: hit.key })
        if (hit.kind !== s.kind) {
          s.setKind(hit.kind)
          // **Crossing closes this.** The button that owns the popover belongs
          // to the console you just left, so leaving it up parks the other
          // discipline's controls — a CFG field over a model that has no CFG —
          // at whatever coordinates its anchor used to have. Same rule the
          // cast picker learned: a picker that stays open on the choice you
          // just made is one that will not take yes for an answer.
          onCross()
        }
      }}>
        {list.map((x) => (
          <option key={`${x.kind}:${x.key}`} value={`${x.kind}:${x.key}`} disabled={!x.ready}>
            {x.label}{x.ready ? '' : ' — missing'}
          </option>
        ))}
      </select>
    </Row>
  )
}

/**
 * The Playground toggle — Default, or a saved workflow run in the console's
 * place, with the compiled prompt fed to the nodes it inherited.
 *
 * Rendered only when workflows exist: a control announcing an absent
 * capability is banned, so every install without a saved workflow never sees
 * this row and the menu is byte-identical to before the feature. The adaptive
 * rows under it are the workflow's own exposed inputs — controls the console
 * cannot name in advance, grown from the save-time diff (see `diffExposes`).
 */
function WorkflowRows({ kind }: { kind: 'image' | 'video' }) {
  const s = useStore()
  const [rows, setRows] = useState<Workflow[] | null>(null)
  useEffect(() => {
    let dead = false
    void listWorkflows().then((r) => {
      if (!dead && !failed(r)) setRows(r.workflows)
    })
    return () => { dead = true }
  }, [])
  const comp = kind === 'image' ? s.img : s.vid
  const set = kind === 'image' ? s.setImg : s.setVid
  // No saved workflows and nothing toggled: the row does not exist. A stale
  // selection with the list gone still renders, because a toggle you cannot
  // reach is a run you cannot explain.
  if (!rows?.length && !comp.workflow) return null
  const current = rows?.find((w) => w.name === comp.workflow)
  const setExtra = (ex: WorkflowExpose, value: string | number | boolean) => {
    set({
      workflowExtras: {
        ...comp.workflowExtras,
        [ex.node]: { ...(comp.workflowExtras[ex.node] ?? {}),
                     [ex.input]: value },
      },
    })
  }
  return (
    <>
      <Row label="Workflow"
           hint={comp.workflow
             ? 'Your saved graph runs instead of the built-in one; the '
               + 'console feeds the nodes it inherited.'
             : undefined}>
        <select value={comp.workflow} onChange={(e) => {
          const name = e.target.value
          const wf = rows?.find((w) => w.name === name)
          const extras: Record<string,
            Record<string, string | number | boolean>> = {}
          for (const ex of wf?.exposes ?? []) {
            const d = ex.default
            ;(extras[ex.node] ??= {})[ex.input] =
              typeof d === 'number' || typeof d === 'boolean' ? d : String(d ?? '')
          }
          set({ workflow: name, workflowExtras: extras })
        }}>
          <option value="">Default</option>
          {(rows ?? []).map((w) => (
            <option key={w.name} value={w.name}>{w.name}</option>
          ))}
          {comp.workflow && !current && (
            <option value={comp.workflow}>{comp.workflow} — missing</option>
          )}
        </select>
      </Row>
      {(current?.exposes ?? []).map((ex) => {
        const val = comp.workflowExtras[ex.node]?.[ex.input]
        return (
          <Row key={`${ex.node}.${ex.input}`} label={ex.label}>
            {ex.type === 'choice' ? (
              <select value={String(val ?? '')}
                      onChange={(e) => setExtra(ex, e.target.value)}>
                {(ex.options ?? []).map((o) => <option key={o}>{o}</option>)}
              </select>
            ) : ex.type === 'toggle' ? (
              <input type="checkbox" checked={!!val}
                     onChange={(e) => setExtra(ex, e.target.checked)} />
            ) : ex.type === 'number' ? (
              <input autoComplete="off" type="number" value={val === undefined ? '' : String(val)}
                     onChange={(e) => {
                       const n = parseFloat(e.target.value)
                       setExtra(ex, Number.isFinite(n) ? n : 0)
                     }} />
            ) : (
              <input autoComplete="off" type="text" value={String(val ?? '')}
                     onChange={(e) => setExtra(ex, e.target.value)} />
            )}
          </Row>
        )
      })}
    </>
  )
}

export function ImageSampling() {
  const s = useStore()
  const pop = usePopover()
  // `?? {}` would widen to `{}` and lose both fields, which is how a `d.steps` that
  // typechecks against nothing gets written. Named, so the two reads below are typed.
  const d: Partial<{ steps: number; cfg: number }> =
    s.state?.krea2_defaults?.[s.img.model] ?? {}
  // Turbo first, against the catalogue's order: the catalogue is ordered by what
  // trains, this picker is read by what generates, and those disagree. Eight steps
  // against twenty-eight is the whole difference, so falling through to RAW because
  // it happens to be listed first is a picker that charges you three and a half
  // times the sampling to open the page.
  const edited = !!(s.img.steps || s.img.cfg || s.img.shift || s.img.sampler || s.img.scheduler)
  const label = s.state?.models.find((m) => m.key === s.img.model)?.label ?? 'Sampling'

  return (
    <>
      <button className={`opt ib${edited ? ' edited' : ''}${pop.open ? ' on' : ''}`}
              id="g-sampling" type="button" title="Sampler, steps and guidance"
              onClick={pop.toggle}>
        {label}
      </button>
      {pop.open && (
        <Popover anchor={pop.anchor} className="menu form" onClose={pop.close}>
          <EngineRow current={s.img.model} onCross={pop.close} />
          <WorkflowRows kind="image" />
          <Row label="Sampler">
            <select value={s.img.sampler || s.state?.image_defaults.sampler || ''}
                    onChange={(e) => s.setImg({ sampler: e.target.value })}>
              {(s.state?.samplers ?? []).map((x) => <option key={x}>{x}</option>)}
            </select>
          </Row>
          <Row label="Scheduler">
            <select value={s.img.scheduler || s.state?.image_defaults.scheduler || ''}
                    onChange={(e) => s.setImg({ scheduler: e.target.value })}>
              {(s.state?.schedulers ?? []).map((x) => <option key={x}>{x}</option>)}
            </select>
          </Row>
          <Row label="Steps">
            <NumInput value={s.img.steps} inputMode="numeric" base={d.steps}
                      placeholder={d.steps != null ? String(d.steps) : 'auto'}
                      onValue={(steps) => s.setImg({ steps })} />
          </Row>
          <Row label="CFG">
            <NumInput value={s.img.cfg} fine={0.1} bigStep={1} base={d.cfg}
                      placeholder={d.cfg != null ? String(d.cfg) : 'auto'}
                      onValue={(cfg) => s.setImg({ cfg })} />
          </Row>
          <Row label="Shift" hint={SHIFT_HINT}>
            <NumInput value={s.img.shift} fine={0.05} bigStep={0.5} placeholder="1.15"
                      onValue={(shift) => s.setImg({ shift })} />
          </Row>
          <Row label="Seed" hint={SEED_HINT}>
            {/* Demoted out of the strip. A seed is *reused off a result* — the
                gesture happens after a render, not before — so by frequency it
                belongs here, and no special pleading about scope is required. */}
            <NumInput value={s.img.seed} inputMode="numeric" placeholder="random"
                      onValue={(seed) => s.setImg({ seed })} />
          </Row>
          <Row label="Images">
            <select value={String(s.img.n)} onChange={(e) => s.setImg({ n: Number(e.target.value) })}>
              {[1, 2, 3, 4].map((n) => <option key={n}>{n}</option>)}
            </select>
          </Row>
          <button className="sz-reset" type="button"
                  onClick={() => {
                    s.setImg({ steps: '', cfg: '', shift: '', seed: '', sampler: '', scheduler: '' })
                    pop.close()
                  }}>
            Reset to the model’s defaults
          </button>
        </Popover>
      )}
    </>
  )
}

export function VideoSampling() {
  const s = useStore()
  const pop = usePopover()
  const m = videoModel(s)
  const r = resolveVid(s)

  return (
    <>
      <button className={`opt ib${vidEdited(s.vid) ? ' edited' : ''}${pop.open ? ' on' : ''}`}
              id="v-sampling" type="button" title="Sampler, steps and guidance"
              onClick={pop.toggle}>
        {m?.label ?? 'Sampling'}
      </button>
      {pop.open && (
        <Popover anchor={pop.anchor} className="menu form" onClose={pop.close}>
          <EngineRow current={s.vid.model} onCross={pop.close} />
          <WorkflowRows kind="video" />
          <Row label="Sampler">
            <select value={r.sampler} onChange={(e) => s.setVid({ sampler: e.target.value })}>
              {(m?.samplers ?? []).map((x) => <option key={x}>{x}</option>)}
            </select>
          </Row>
          <Row label="Scheduler">
            <select value={r.scheduler} onChange={(e) => s.setVid({ scheduler: e.target.value })}>
              {(m?.schedulers ?? []).map((x) => <option key={x}>{x}</option>)}
            </select>
          </Row>
          <Row label="Steps">
            <NumInput value={s.vid.steps} inputMode="numeric" base={m?.defaults.steps}
                      placeholder={r.stepsPlaceholder}
                      onValue={(steps) => s.setVid({ steps })} />
          </Row>
          {/* CFG, Shift and Expert switch stood here, drawn only for a model
              that read them — "a row the model does not read is not drawn". The
              only video family is guidance-distilled with one expert, so all
              three were permanently undrawn, and a conditional whose condition
              is a constant is a branch kept for a model that is not there. */}
          <Row label="Seed" hint={SEED_HINT}>
            <NumInput value={s.vid.seed} inputMode="numeric" placeholder="random"
                      onValue={(seed) => s.setVid({ seed })} />
          </Row>
          <button className="sz-reset" type="button"
                  onClick={() => {
                    s.setVid({ steps: '', seed: '', sampler: '', scheduler: '' })
                    pop.close()
                  }}>
            Reset to the model’s defaults
          </button>
        </Popover>
      )}
    </>
  )
}
