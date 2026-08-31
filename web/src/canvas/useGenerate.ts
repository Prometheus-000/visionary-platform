import { useCallback, useRef, useState } from 'react'

import { everyMs, failed, type ApiError } from '../api/client'
import { generate, status, stop } from '../api/routes'
import type { JobStatus } from '../api/types'
import type { GalleryItem } from '../gallery/types'
import { loraIndex, readChips, stripLoras } from '../lora/tokens'
import { readRegions } from '../regions/geometry'
import { refreshArsenal } from '../scene/arsenal'
import { attached, negAllowed, readShot, regionsLive, useStore, type Store } from '../store'
import { readSize } from '../console/size'

/**
 * One render, from press to pictures.
 *
 * Polls through `everyMs` at 400ms. That interval is why `everyMs` exists at all:
 * `/api/status` reads a *network* Dict, and `setInterval` fires on a clock rather
 * than on a reply, so a slow answer does not delay the next tick, it overlaps it.
 * Replies then land out of order and the bar is painted by whichever arrived last
 * rather than whichever is newest — step 14 lands, step 12 lands on top of it, and
 * the bar walks backwards.
 */
export type RunState = {
  running: boolean
  /** The render on screen — the last one that *completed*. It is deliberately not
   *  cleared when the next run starts: a new render replaces the old one when it
   *  lands, not when it is asked for, so an iteration you are judging stays up
   *  through the two minutes it takes to make its successor. `finish` is the only
   *  thing that moves it. */
  jobId: string | null
  files: string[]
  /** The job being polled right now, which is *not* `jobId` until it completes.
   *  Two ids because the bytes on screen and the work in flight are two different
   *  renders during a run — `fileUrl(jobId, f)` has to keep addressing the old
   *  job's files while `runId` is the one `/api/status` is asked about. */
  runId: string | null
  percent: number
  phase: string
  /** The whole `ApiError` where there is one, not just its sentence — `ErrorNote`
   *  puts the server's own report in a closed disclosure under it, and narrowing
   *  to `.error` here is what threw that away before it reached the box. */
  error: string | ApiError | null
  /** The facts about the render, in the page's order: seeds, then the sampler line,
   *  then how long it took. */
  meta: string[]
  /** LoRAs the run reports it could not apply, each already phrased as advice —
   *  see `skipNote`. */
  skipped: string[]
  /** What the render on screen was asked for, so the `<img>` can carry `width`/`height`
   *  and the browser can reserve the box from the ratio before a byte of it arrives.
   *  Without them the canvas column reflows at the moment the picture lands — which is
   *  the moment a hand is over the strip underneath it.
   *
   *  Paired with `jobId`/`files` and moved only by `finish`, for the same reason those
   *  are: during a run the *previous* render is still on screen, so reserving from the
   *  size box would describe the picture being made rather than the one being shown, and
   *  changing the ratio a control at a time would shift the thing this exists to hold
   *  still. `/api/status` does not report dimensions, so these come from the request.
   *  0 means unknown — a run restored from a reload has no request to read. */
  w: number
  h: number
}

/**
 * A skip reason, turned into the thing to do about it.
 *
 * The server's own words go straight to the canvas, and its words are diagnostic:
 * `no matching keys` is precise, unactionable, and — when the LoRA is named
 * something that reads as English — collides with the name into a line nobody can
 * parse. `not applied: gone (no matching keys)` was observed live and there is no
 * reading of that sentence that tells you it is a base-model mismatch.
 *
 * So the name is quoted, which is what stops it being read as a word, and the one
 * reason this actually emits is translated into its cause. Anything else the run
 * invents still passes through verbatim: an unknown reason said plainly beats a
 * guess dressed as advice.
 */
function skipNote(name: string, reason?: string): string {
  const who = name ? `"${name}"` : 'a LoRA'
  const r = (reason ?? '').trim()
  // No full stops on any of these: they are joined into one line by the caller, and a
  // sentence that already ends gets a comma welded to it.
  if (!r) return `${who} — the run did not say why; check it is the file you meant`
  if (/matching keys/i.test(r)) {
    return `${who} — its keys don’t match this model, so it was likely trained for a`
      + ' different base'
  }
  return `${who} — ${r}`
}

const IDLE: RunState = {
  running: false, jobId: null, files: [], runId: null, percent: 0, phase: '',
  error: null, meta: [], skipped: [], w: 0, h: 0,
}

/**
 * The `/api/generate` body.
 *
 * One function, so the Generate button, ⌘Enter and anything Phase 6 grows later all
 * send the same request — which is what lets the console be *dissolvable*: Phase 6
 * re-hosts these controls onto the canvas and the payload does not notice.
 *
 * Empty strings are passed through rather than coerced: the route's `num()` treats
 * `''` as "use the default" and would take a `0` literally, so turning a blank steps
 * field into a number here would silently ask for zero steps.
 */
export function imageBody(s: Store): Record<string, unknown> {
  const index = loraIndex(s.state)
  const [width, height] = readSize(s.img)
  const regions = readRegions(index, s.regions, regionsLive(s))
  const prompt = stripLoras(s.prompt)
  return {
    prompt,
    negative_prompt: negAllowed(s) ? s.negative : '',
    model: s.img.model,
    shot: readShot(s.shot),
    loras: readChips(s.loras, s.state?.max_loras ?? 6),
    regions,
    region_weight: s.regionWeight,
    // Only when there are boxes to compose around — the backend rejects a plate
    // without regions, and sending one anyway would turn a hidden tile that still
    // holds an image into an error nobody could see the cause of.
    scene: regions.length ? attached(s.frame, 'scene') : null,
    outfit: regions.length ? attached(s.frame, 'outfit') : null,
    // Sockets 3 and 4 — a photo and the user's sentence about it, compacted
    // so removing the first object does not send a hole.
    // Style rides with or without boxes — it is the no-boxes engine — and the
    // route answers the conflicting pair with a form error the note has
    // already explained.
    style_refs: [attached(s.frame, 'style1')].filter(Boolean),
    style_strength: s.styleStrength,
    objects: regions.length
      ? (['object1', 'object2'] as const)
          .map((role) => s.frame.attachments.find((a) => a.role === role))
          .filter(Boolean)
          .map((a) => ({ image: a!.image, note: (a!.note ?? '').trim() }))
      : null,
    width,
    height,
    num_images: s.img.n,
    seed: s.img.seed,
    sampler: s.img.sampler,
    scheduler: s.img.scheduler,
    steps: s.img.steps,
    cfg_scale: s.img.cfg,
    shift: s.img.shift,
    gpu: s.gpu.image,
    // The Playground toggle. Only while set — a request that names no
    // workflow must be byte-identical to one from before the feature.
    ...(s.img.workflow && {
      workflow: s.img.workflow,
      workflow_extras: s.img.workflowExtras,
    }),
  }
}

export function useGenerate(onLanded: (it: GalleryItem) => void) {
  const [run, setRun] = useState<RunState>(IDLE)
  /** The size the in-flight run was asked for, waiting for `finish` to promote it
   *  alongside that run's files. A ref rather than state: nothing paints from it until
   *  it lands, so a render for it would be a render for nothing. */
  const asked = useRef<{ w: number; h: number } | null>(null)

  const finish = useCallback((s: JobStatus, jobId: string) => {
    const files = (s.files as string[] | undefined) ?? []

    // `=== false`, not `!l.applied`. The image report emits {name, unet,
    // text_encoder} and no `applied` key at all, so a falsy test called every LoRA on
    // every render unapplied — a warning that is always on is worse than no warning,
    // and this one cost real time during a debug by pointing at a healthy LoRA while
    // the actual fault was elsewhere. Both other readers of this field already
    // default the other way.
    const loras =
      (s.loras as { name?: string; applied?: boolean; reason?: string }[] | undefined) ?? []
    // The reason rides along, phrased as the fix rather than as the symptom — see
    // `skipNote` for what "no matching keys" cost when it was printed raw.
    const skipped = loras
      .filter((l) => l.applied === false)
      .map((l) => skipNote(String(l.name ?? ''), l.reason))

    // Field names are the record's, not ones that sound right: `seeds` is a list
    // because a batch has one per image, and the duration is `duration_s`.
    const seeds = (s.seeds as number[] | undefined) ?? []
    const meta = [
      seeds.length ? `seed ${seeds.join(', ')}` : '',
      s.sampler ? `${String(s.sampler)} · ${String(s.steps)} steps · CFG ${String(s.cfg_scale)}` : '',
      s.duration_s ? `${String(s.duration_s)}s` : '',
    ].filter(Boolean)

    // **Nothing writes the seed back into the box, and that is the decision
    // rather than the gap.** A field that silently stopped being random reads
    // as "Generate is broken, it keeps making the same picture" — so the seed
    // rolls until you type one, and the one that made a picture you liked is
    // on that picture, one Reuse away.
    // Replaces the on-screen render atomically: the new job's id and files land in
    // the same set as `running:false`, so there is never a frame where the old
    // `jobId` is paired with the new `files`.
    setRun((p) => ({
      ...p, running: false, jobId, runId: null, files, percent: 100, phase: '',
      error: null, meta, skipped,
      // With the files, never before them: these describe the picture that is landing.
      w: asked.current?.w ?? 0, h: asked.current?.h ?? 0,
    }))
    // The finished run, handed over rather than thrown away. The gallery used to ask
    // the volume to tell it about a job this function was holding in its hand, and
    // believe the answer when it came back without one. `**s` first so the four facts
    // below win — the record is the run's report, and these are what the card draws.
    onLanded({ ...(s as Partial<GalleryItem>), job_id: jobId, kind: 'image', files,
               created: Date.now() / 1000 })
  }, [onLanded])

  const start = useCallback(async () => {
    // The image side's half of the same rule — see `useVideo.start`. A box
    // holding a recalled character re-reads it off the shelf before the body is
    // built, so the face that renders is the one in the library now.
    await refreshArsenal()
    const s = useStore.getState()
    const body = imageBody(s)
    if (!body.prompt && !(body.regions as unknown[]).length) return
    // Read off the body rather than the store, so it is the size actually being asked
    // for — the two differ, because `readSize` floors and clamps what the box holds.
    asked.current = { w: Number(body.width) || 0, h: Number(body.height) || 0 }
    // Keep the previous render (jobId/files/meta) on screen and only overlay a
    // progress state on it — see the `jobId` note above. A cold first run has no
    // previous render, so this shows the full placeholder; an iteration keeps the
    // last picture up until its replacement is ready.
    setRun((p) => ({
      ...p, running: true, runId: null, percent: 0, phase: 'Queued…', error: null,
    }))
    const r = await generate(body)
    if (failed(r)) {
      // Leave the last render up; a failed request should not blank what you were
      // iterating on. Only the error and the stopped-running state change. The
      // whole `ApiError` is kept, not `r.error` — the traceback is the only thing
      // in the failure that says what actually broke, and `ErrorNote` has a place
      // to put it now.
      setRun((p) => ({ ...p, running: false, runId: null, error: r }))
      return
    }
    const runId = r.job_id
    setRun((p) => ({ ...p, runId }))

    const t = everyMs(async () => {
      const st = await status(runId)
      if (failed(st)) return
      if (st.status === 'completed') {
        clearInterval(t)
        finish(st, runId)
      } else if (st.status === 'failed') {
        clearInterval(t)
        // The job's own message when it has one. When it does not, "Generation
        // failed" was the whole of what the page said — a restatement of the one
        // thing the absent picture already tells you, with nothing to do next. The
        // two things that are actually knowable from here are that the request got
        // as far as a job (so the deployment is up) and which of them a person can
        // change without leaving the page.
        setRun((p) => ({
          ...p, running: false, runId: null,
          error: st.error
            || 'The render failed and the job gave no reason — press Generate to try'
               + ' again, and check the model and GPU under Settings if it keeps failing.',
        }))
      } else if (st.status === 'stopped') {
        clearInterval(t)
        // The previous render coming back is the feedback now — a stopped run leaves
        // the picture it was replacing exactly where it was, so there is nothing to
        // say that the returned image does not already say.
        setRun((p) => ({ ...p, running: false, runId: null, phase: '' }))
      } else {
        setRun((prev) => ({
          ...prev,
          running: true,
          percent: Number(st.percent ?? 0),
          // The step count when there is one: "Working…" is equally true of a run on
          // step 2 and one that has stalled. And every phase before the first
          // step is shown as the server named it — "reloading the volume",
          // "staging the inputs" — because those are where the minutes go on a
          // cold container and "Working…" over all of them is the state
          // somebody kills the app out of.
          phase: st.step
            ? `Step ${st.step}/${String(st.total_steps ?? st.steps ?? '?')}`
            : (st.phase === 'loading' ? 'Loading the model…' : (st.phase || 'Working…')),
        }))
      }
    }, 400)
  }, [finish])

  /** Cooperative: the job checks a flag between steps and unwinds cleanly, so the
   *  container survives and the next request is warm. Disabled rather than removed on
   *  press, because a Stop that vanishes on click reads as a click that missed. */
  const cancel = useCallback(async () => {
    if (!run.runId) return
    setRun((p) => ({ ...p, phase: 'Stopping…' }))
    await stop(run.runId)
  }, [run.runId])

  const clear = useCallback(() => setRun(IDLE), [])

  return { run, start, cancel, clear }
}
