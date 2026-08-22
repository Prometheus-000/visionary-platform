/**
 * Putting a finished run back in the composer.
 *
 * The point of keeping the sidecar: everything here reproduces the result, so this is a
 * read of a file rather than a re-typed prompt.
 *
 * **The pills come back, not the sentence they compiled into.** A run whose prompt was a
 * six-field document has to be restored as the rail that built it — putting the document
 * into the prompt box would put a schema in a textarea and then compile *that* on the
 * next run, which is the one way this feature could corrupt a reuse. `shot` is only in
 * the sidecar when the compiler did something, so a card from before the palette existed
 * clears the rail rather than keeping whatever happened to be on it.
 */
import { BUCKETS, commitBoxes } from '../console/size'
import { chipFrom, loraChips, loraIndex, resolveLora, stripLoras } from '../lora/tokens'
import { newRegion, shotItem, useStore } from '../store'
import type { LoraChip } from '../lora/tokens'
import type { GalleryItem } from './types'

const set = (v: unknown): string => (v == null || v === '' ? '' : String(v))

/**
 * Put the prompt back, and the document with it — keyed to that exact string.
 *
 * **The one place this feature could quietly corrupt a run**, which is why the
 * key is *computed from what was just written* rather than from the field the
 * document came out of. Those are the same string today, and a document keyed
 * to a plausible-looking near-miss is a document `docFor` refuses forever: the
 * marks never appear, the run silently goes plain, and nothing on screen says
 * why. Deriving both from one value is what makes that unreachable instead of
 * merely unlikely.
 *
 * `stripLoras`, because that is the form the interpreter was given and the form
 * `/api/generate` receives — the tokens ride in the box and never in the prose.
 */
/** The prompt and the chips are two fields again, so they are restored as two.
 *  They were one string while a LoRA was `<lora:…>` in the prose; a chip is not
 *  text and never belonged in there. */
function restore(s: ReturnType<typeof useStore.getState>,
                 typed: string, chips: LoraChip[], modules: GalleryItem['modules'],
                 original = ''): void {
  const written = typed
  s.setPrompt(written)
  useStore.setState({ loras: chips })
  const prose = stripLoras(written)
  s.setDoc(modules?.length
    ? { for: prose, from: original || prose, elements: modules, text: prose }
    : null)
}

export function reuse(it: GalleryItem): void {
  const s = useStore.getState()
  const index = loraIndex(s.state)
  const vocab = s.state?.shot_vocab ?? []

  // **The kind first, and that ordering is load-bearing.** `setKind` stashes the
  // composer it is leaving and installs the other one — prompt, negative, pills
  // and chips together — so anything written before it is written into the buffer
  // about to be put away.
  s.setKind(it.kind === 'image' ? 'image' : 'video')
  s.setShot((it.shot ?? []).filter((p) => p && shotItem(vocab, p.key)))
  s.setShotOpen(null)
  s.setRefRoles((it.ref_roles ?? []).slice())

  if (it.kind === 'image') {
    // A LoRA deleted since the run simply does not come back — a chip is picked
    // from a list and always resolves, so there is no such thing as one naming
    // nothing.
    restore(s, it.prompt_typed || it.prompt || '',
            loraChips(index, it.loras, false), it.modules)
    s.setNegative(it.negative_prompt ?? '')

    const size = `${it.width}x${it.height}`
    if (BUCKETS.some(([v]) => v === size)) {
      s.setImg({ aspect: size, scale: '1', width: it.width!, height: it.height! })
    } else if (it.width && it.height) {
      // Not a preset, so it was a custom size — and it has to come back as one, or
      // "reuse" would silently render a different picture than the card.
      s.setImg(commitBoxes({ ...s.img, aspect: 'custom', width: it.width, height: it.height }))
    }
    s.setImg({
      ...(it.model ? { model: it.model } : {}),
      // `seed` is what this backend records; `seeds` is what the Forge one did, and
      // sidecars written by it are still on the volume. Reading both keeps every card in
      // the gallery reusable rather than only the ones made since.
      seed: set(it.seed ?? it.seeds?.[0]),
      sampler: set(it.sampler),
      scheduler: set(it.scheduler),
      steps: set(it.steps),
      cfg: set(it.cfg_scale),
      shift: set(it.shift),
    })

    // Regions come back or the reuse is a lie: which LoRA sat in which box is the whole
    // result of a regional render, and restoring the prompt without them would put a
    // one-subject picture behind a card showing two people.
    //
    // The plates deliberately do not come back — they were uploaded bytes, not something
    // the sidecar keeps, and the note under the stack will say the run is the one-pass
    // kind until one is dropped again.
    const saved = it.regions ?? []
    s.setRegions(saved.map((r) => {
      // `r.lora` is the path relative to loras/, which is what `resolveLora` matches
      // first — reducing it to a stem here would break exactly the ambiguous names the
      // full path exists to disambiguate.
      const hit = r.lora && r.lora !== 'None' ? resolveLora(index, r.lora) : null
      const box = r.box ?? []
      return newRegion({
        prompt: r.prompt ?? '',
        lora: hit ? { ...chipFrom(hit, true), strength: r.strength ?? 1.3 } : null,
        // The sidecar records whether a box had a photo, never the photo — it was
        // uploaded bytes staged into a container that is long gone.
        attachments: [],
        x: Number(box[0]) || 0,
        y: Number(box[1]) || 0,
        w: box[2] != null ? Number(box[2]) : 1,
        h: box[3] != null ? Number(box[3]) : 1,
      })
    }))
    s.select(saved.length ? 0 : -1)
    if (it.region_weight != null) s.setRegionWeight(String(it.region_weight))
    // Reuse restores boxes onto an empty canvas, so they are meant to be seen and
    // adjusted — that is the geometry surface, not a render to keep clean.
    s.setEdit('geometry')
    s.attach('frame', 'scene', null)
    s.attach('frame', 'outfit', null)
    return
  }

  // The model first, because it decides which controls exist: restoring a CFG into a
  // panel built for a family that reads none is a value written to a box that is about
  // to be hidden.
  const model = s.state?.video_models.find((m) => m.key === it.model && m.ready)
  s.setVid({
    ...(model ? { model: model.key } : {}),
    seed: set(it.seed),
    steps: set(it.steps),
    cfg: set(it.cfg_scale),
    shift: set(it.shift),
    switchAt: set(it.switch_at),
    // Explicit rather than left blank, so restoring a clip's settings and then attaching
    // a frame does not quietly hand them back to the model's defaults.
    sampler: set(it.sampler),
    scheduler: set(it.scheduler),
    seconds: it.seconds ? String(Math.round(it.seconds)) : '',
    ...(it.width && it.height && !s.keyframe.first
      ? { aspect: nearestVideoAspect(it.width / it.height) }
      : {}),
  })
  // Matched on the full filename under loras/, not the stem: `high.safetensors` is the
  // filename of both speed pairs, so a stem match would restore the t2v LoRA into an
  // i2v run without a word about it.
  restore(s, it.prompt_typed || it.prompt || '',
          loraChips(index, it.loras, true), it.modules)
  s.setNegative(it.negative_prompt ?? '')
}

const VIDEO_ASPECTS = ['21:9', '16:9', '4:3', '1:1', '3:4', '9:16']

function nearestVideoAspect(r: number): string {
  return VIDEO_ASPECTS
    .map((v) => {
      const [a, b] = v.split(':').map(Number)
      return { v, d: Math.abs((a ?? 1) / (b ?? 1) - r) }
    })
    .sort((x, y) => x.d - y.d)[0]?.v ?? '16:9'
}
