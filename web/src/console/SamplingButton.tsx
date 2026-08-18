import { Popover, usePopover } from '../ui/Popover'
import { NumInput } from '../ui/NumInput'
import { supports, useStore, videoModel } from '../store'
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
// dimension boxes get: a field that silently stopped being random is a value
// that cannot show itself. There is no lock glyph, no dice and no chip —
// the number is visible in its own field and one gesture from gone, which is
// what makes saying so enough.
const SEED_HINT = 'Blank draws a new one. A seed worth keeping is on the render '
  + 'that used it — and once the prompt has been read, the first one is kept here '
  + 'so an edit changes only what it implied. Clear it to roll again.'
const SHIFT_HINT = 'Bends the noise schedule — higher spends more steps on composition and motion.'

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
  const models = ['turbo', 'raw']
    .map((k) => s.state?.models.find((m) => m.key === k))
    .filter((m): m is NonNullable<typeof m> => !!m)
  const edited = !!(s.img.steps || s.img.cfg || s.img.shift || s.img.sampler || s.img.scheduler)
  const label = models.find((m) => m.key === s.img.model)?.label ?? 'Sampling'

  return (
    <>
      <button className={`opt ib${edited ? ' edited' : ''}${pop.open ? ' on' : ''}`}
              id="g-sampling" type="button" title="Sampler, steps and guidance"
              onClick={pop.toggle}>
        {label}
      </button>
      {pop.open && (
        <Popover anchor={pop.anchor} className="menu form" onClose={pop.close}>
          <Row label="Model">
            <select value={s.img.model} onChange={(e) => s.setImg({ model: e.target.value })}>
              {models.map((m) => (
                <option key={m.key} value={m.key} disabled={!m.present}>
                  {m.label}{m.present ? '' : ' — missing'}
                </option>
              ))}
            </select>
          </Row>
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
  const sup = supports(s)
  const models = s.state?.video_models ?? []

  return (
    <>
      <button className={`opt ib${vidEdited(s.vid) ? ' edited' : ''}${pop.open ? ' on' : ''}`}
              id="v-sampling" type="button" title="Sampler, steps and guidance"
              onClick={pop.toggle}>
        {m?.label ?? 'Sampling'}
      </button>
      {pop.open && (
        <Popover anchor={pop.anchor} className="menu form" onClose={pop.close}>
          <Row label="Model">
            <select value={s.vid.model} onChange={(e) => s.setVid({ model: e.target.value })}>
              {models.map((v) => (
                <option key={v.key} value={v.key} disabled={!v.ready}>
                  {v.label}{v.ready ? '' : ' — missing'}
                </option>
              ))}
            </select>
          </Row>
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
          {/* A row the model does not read is not drawn. H3 is guidance-distilled,
              so a CFG box on it is a promise the sampler will not keep. */}
          {sup.cfg && (
            <>
              <Row label="CFG">
                <NumInput value={s.vid.cfg} fine={0.1} bigStep={1} base={m?.defaults.cfg}
                          placeholder={r.cfgPlaceholder}
                          onValue={(cfg) => s.setVid({ cfg })} />
              </Row>
              <Row label="Shift" hint={SHIFT_HINT}>
                <NumInput value={s.vid.shift} fine={0.1} bigStep={1} base={m?.defaults.shift}
                          placeholder={r.shiftPlaceholder}
                          onValue={(shift) => s.setVid({ shift })} />
              </Row>
            </>
          )}
          {sup.experts && (
            <Row label="Expert switch"
                 hint="Step at which the high-noise expert hands the latent to the low-noise one.">
              <NumInput value={s.vid.switchAt} inputMode="numeric" bigStep={4} placeholder="auto"
                        onValue={(switchAt) => s.setVid({ switchAt })} />
            </Row>
          )}
          <Row label="Seed" hint={SEED_HINT}>
            <NumInput value={s.vid.seed} inputMode="numeric" placeholder="random"
                      onValue={(seed) => s.setVid({ seed })} />
          </Row>
          <button className="sz-reset" type="button"
                  onClick={() => {
                    s.setVid({ steps: '', cfg: '', shift: '', switchAt: '', seed: '',
                               sampler: '', scheduler: '' })
                    pop.close()
                  }}>
            Reset to the model’s defaults
          </button>
        </Popover>
      )}
    </>
  )
}
