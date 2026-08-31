/**
 * Every route the page can call, named once.
 *
 * The vanilla page built these paths at 13 call sites with template literals,
 * which is fine until a name changes and the compiler has nothing to say about
 * it. Listing them here is not ceremony — it is the only thing that makes the
 * server and the front end fail together rather than at runtime, now that the
 * two are separate builds.
 *
 * `encodeURIComponent` on every interpolated segment, without exception. A
 * dataset called "shoot 2 / picks" and a LoRA folder with a space in it are
 * both ordinary, and the routes take them as path parameters.
 */
import { api, post, type Res } from './client'
import type {
  AppState, CompileResult, DupeReport, Insight, JobStatus, Session, ShotPill } from './types'

const seg = encodeURIComponent

/* ---- state, settings, weights ---------------------------------------- */

export const getState = () => api<AppState>('/api/state')
export const getWhere = () => api<Record<string, unknown>>('/api/where')
/** `hf_token`, which is the field `set_token` reads. It was `token` here, and
 *  the route answers `{ok: true, hf_token_set: false}` to that — a 200 that
 *  cleared the token it was asked to save, so Save reported success and the
 *  gated downloads went on failing with no visible connection between the two. */
export const setToken = (hfToken: string) =>
  post<{ ok?: boolean; hf_token_set?: boolean }>('/api/token', { hf_token: hfToken })

/**
 * What a download route answers with.
 *
 * `mine` is the half that is easy to miss. One download at a time, because they
 * share an uplink — three concurrent pulls measured 4-12 MB/s each against
 * ~31 MB/s for one, so a second download is the same bandwidth divided plus a
 * container to pay for. So `/api/download` is *idempotent*: a second press
 * returns the job the first one started, with `mine: false` and `busy_with`
 * naming what holds it. Being busy is a state, not an error — the page disables
 * the other buttons rather than answering a press with a red box, because
 * pressing twice is what anyone does when the first press appears to do nothing.
 */
export type DownloadStart = {
  job_id?: string
  mine?: boolean
  busy_with?: string
  note?: string
}

/** One weight, by catalogue key. Its job id is `dl_{key}`. */
export const startDownload = (key: string) => post<DownloadStart>('/api/download', { key })

/**
 * Every missing file in one family, queued server-side.
 *
 * The family is the unit you decide in — you want the Wan stack or you do
 * not — and clicking its files one at a time meant watching a 4 GB file finish
 * to be allowed to start the next, which is a queue kept in a person rather
 * than in the program. The token rides along because pasting a key and pressing
 * Download is one action, and the gated families are exactly the ones you paste
 * it for.
 *
 * There is deliberately no catalogue-wide form of this. It existed, in the
 * token row, which put the one button that pulls every family — including the
 * ones an install will never run — next to a password field it has nothing to
 * do with.
 */
export const downloadFamily = (family: string, hfToken?: string) =>
  post<DownloadStart>('/api/download-missing', { family, hf_token: hfToken || '' })

/** `refetch` turns the diff off. A folder is listed before it is fetched and a
 *  name already in `loras/` is left alone, which is what makes re-pasting a link
 *  after two more epochs were uploaded cost the two epochs — but the listing
 *  carries no size and no checksum, so a file replaced in place under the name it
 *  already had can only be picked up by asking for the lot again. */
export const startGdrive = (url: string, folder?: string, refetch?: boolean) =>
  post<DownloadStart>('/api/gdrive', { url, folder, refetch })
/**
 * One LoRA — the folder with its epochs in it, or the loose file. Singular
 * `path`, and the unit is the row `/api/state` lists, which is the unit the
 * storage layout already calls a LoRA.
 *
 * An epoch inside a folder is deliberately not addressable: the route rejects
 * `loras/{name}/{epoch}.safetensors` with the same guard that rejects `../`
 * escapes, because one request whose blast radius is a file or a whole training
 * run depending on path depth is the thing that guard exists to prevent.
 */
export const deleteLora = (path: string) => post<{ ok?: boolean }>('/api/loras/delete', { path })

/* ---- session --------------------------------------------------------- */

/** A draft belongs to the window that made it; this is the heartbeat that
 *  keeps it from being swept after fifteen minutes of silence. `open` names
 *  the set on screen, so a draft you are looking at becomes this window's —
 *  liveness follows attention, not authorship. */
export const beat = (session: string, open?: string | null) =>
  post<{ ok?: boolean }>('/api/session', { session, open: open || undefined })

/* ---- datasets and drafts --------------------------------------------- */

export const listDatasets = () => api<Record<string, unknown>>('/api/datasets')
export const createDataset = (body: unknown) => post<Record<string, unknown>>('/api/datasets', body)
export const getDataset = (name: string) => api<Record<string, unknown>>(`/api/datasets/${seg(name)}`)
export const saveDataset = (name: string, body: unknown) =>
  post<Record<string, unknown>>(`/api/datasets/${seg(name)}/save`, body)
export const setDatasetMeta = (name: string, body: unknown) =>
  post<Record<string, unknown>>(`/api/datasets/${seg(name)}/meta`, body)
export const deleteDataset = (name: string) =>
  post<Record<string, unknown>>(`/api/datasets/${seg(name)}/delete`)
export const captionDataset = (name: string, body: unknown) =>
  post<Record<string, unknown>>(`/api/datasets/${seg(name)}/caption`, body)
export const removeImage = (name: string, body: unknown) =>
  post<Record<string, unknown>>(`/api/datasets/${seg(name)}/remove`, body)
/** The trigger word rides in the query, because trigger coverage is the first
 *  thing the panel reports and the word is typed on the page rather than stored
 *  — asking without it answers "0 of 24 have the trigger" for a set that is
 *  fine. */
export const datasetInsight = (name: string, trigger = '') =>
  api<Insight>(`/api/datasets/${seg(name)}/insight?trigger=${seg(trigger)}`)
export const prependTrigger = (name: string, body: unknown) =>
  post<Record<string, unknown>>(`/api/datasets/${seg(name)}/prepend-trigger`, body)
/** Find & replace across caption sidecars. `images` scopes it to the page's
 *  current filtered view — the filters are the targeting tool. */
export const replaceCaptions = (name: string, body: {
  find: string
  replace: string
  match_case: boolean
  images?: string[]
}) => post<{ ok?: boolean; changed?: number }>(`/api/datasets/${seg(name)}/replace`, body)

/* ---- caption presets and models --------------------------------------- */

/** Save the edited instruction as a named preset, kept in the config Dict so
 *  it survives redeploys. Re-saving the same name updates it. */
export const saveCaptionPreset = (label: string, instruction: string) =>
  post<{ ok?: boolean; key?: string }>('/api/caption/presets', { label, instruction })
export const deleteCaptionPreset = (key: string) =>
  post<{ ok?: boolean }>('/api/caption/presets/delete', { key })
/** A captioner by HF repo id. Validated server-side against the repo's own
 *  config.json before it lands in the menu, so a typo or a text-only model is
 *  a form error rather than a cold GPU start. */
export const addCaptionModel = (repo: string, label?: string) =>
  post<{ ok?: boolean; key?: string }>('/api/caption/models', { repo, label: label || '' })
export const deleteCaptionModel = (key: string) =>
  post<{ ok?: boolean }>('/api/caption/models/delete', { key })

/**
 * Duplicate groups at one of three named similarity levels.
 *
 * Its own route rather than a field on `/insight`, because the two are priced
 * differently: insight reads the `.txt` files and is refreshed on every caption
 * edit, while this decodes every image in the folder the first time it runs.
 */
export const datasetDuplicates = (name: string) =>
  api<DupeReport>(`/api/datasets/${seg(name)}/duplicates`)

/** Bytes off the volume, by their own route — never inlined into a polled
 *  record. A dict polled every two seconds must not grow with its result. */
export const thumbUrl = (name: string, file: string) =>
  `/api/thumb/${seg(name)}/${seg(file)}`
export const imageUrl = (name: string, file: string) =>
  `/api/image/${seg(name)}/${seg(file)}`
/** A clip in a set, streamed with Range so a tile can seek to one frame rather
 *  than fetch the file. There is no thumbnail route for one — web_image has no
 *  ffmpeg — so the tile paints its own poster the way a gallery card does. */
export const clipUrl = (name: string, file: string) =>
  `/api/clip/${seg(name)}/${seg(file)}`

/* ---- jobs ------------------------------------------------------------ */

export const caption = (body: unknown) => post<{ job_id: string }>('/api/caption', body)
export const train = (body: unknown) => post<{ job_id: string }>('/api/train', body)

/* ---- training sessions ------------------------------------------------ */

/**
 * The board. One request for every card, live fields included.
 *
 * Not one poll per card: a browser gives an origin about six connections, and
 * the whole point of running four LoRAs at once is that the fifth thing on
 * screen — the contact sheet you are captioning while they train — still
 * loads. The server merges each card with its job record, which it holds
 * anyway.
 */
export const listSessions = () =>
  api<{ sessions: Session[]; total?: number }>('/api/sessions')
/** Create when `id` is absent, save an edit when it is there. The server
 *  refuses an edit to a running card rather than letting the dials and the
 *  process disagree about what is training. */
export const putSession = (body: unknown) =>
  post<{ session: Session }>('/api/sessions', body)
export const startSession = (id: string) =>
  post<{ job_id?: string; session?: Session; already?: boolean }>(`/api/sessions/${seg(id)}/start`)
/** Cooperative, and the card stays with the progress it reached. */
export const stopSession = (id: string) =>
  post<{ session?: Session }>(`/api/sessions/${seg(id)}/stop`)
/** The card goes; anything it started is asked to stop on the way out. What it
 *  wrote into `loras/` is not touched — that is deleted where LoRAs are. */
export const deleteSession = (id: string) =>
  post<{ ok?: boolean }>(`/api/sessions/${seg(id)}/delete`)
export const generate = (body: unknown) => post<{ job_id: string }>('/api/generate', body)
export const video = (body: unknown) => post<{ job_id: string }>('/api/video', body)

export const status = (jobId: string) => api<JobStatus>(`/api/status/${seg(jobId)}`)
/** Cooperative: the job checks a flag between steps and unwinds cleanly, so
 *  the container survives and the next request is warm. */
export const stop = (jobId: string) => post<{ ok?: boolean }>(`/api/stop/${seg(jobId)}`)

/* ---- the compiler ---------------------------------------------------- */

/**
 * The exact document a run would be given.
 *
 * This is the one route the React side must never reimplement. It is the same
 * compiler on the same container as the run, which is what makes the
 * disclosure under the pill rail truthful — a preview with its own
 * implementation is a preview that can disagree with what happens, which is
 * worse than no preview. `tools/ui-checks/probe_compile.py` pins 402 of its
 * outputs for exactly this reason.
 */
export const compile = (body: {
  prompt: string
  shot: ShotPill[]
  kind?: 'image' | 'video'
  model?: string
  seconds?: number
  /** **Counts, not pictures.** The compiler needs only that there *is* a
   *  reference — the bytes would make a route re-fetched on every keystroke
   *  carry megabytes. The scene's own refs are indices into these same counts. */
  references?: number
  ref_videos?: number
  ref_audios?: number
  ref_roles?: string[]
  first_frame?: boolean
  last_frame?: boolean
  /** The cast and the timeline. Validated here as well as on the way to the GPU,
   *  because this is what the composer polls: a handle nobody defined should be
   *  a sentence under the rows the moment it is typed, not a refusal discovered
   *  at Generate. */
  scene?: unknown
}) => post<CompileResult>('/api/compile', body)

/**
 * Start the container this session will use, while the page is still being read.
 *
 * It warmed the interpreter's L4 for a long time after nothing routed there,
 * which cost a cold start on every page load to warm a card no request would
 * reach. It warms the *generator* now, so this window is spent getting ComfyUI
 * up and the checkpoint resident — which is the whole of what the first
 * Generate waits for, since that container loads one model again.
 *
 * Fire and forget: nothing on screen depends on it, and a slow first press is
 * the only cost of it failing.
 */
export const warm = (kind: 'image' | 'video' = 'image') =>
  post<{ ok: boolean }>('/api/warm', { kind })

/* ---- outputs and the gallery ----------------------------------------- */

/** The Arsenal's characters. The listing is names and notes for the picker;
 *  the files come one at a time off their own route when somebody picks. */
export const characters = () =>
  api<Record<string, unknown>>('/api/characters')
export const saveCharacter = (handle: string, body: unknown) =>
  post<Record<string, unknown>>(`/api/characters/${seg(handle)}`, body)
export const characterFileUrl = (handle: string, file: string) =>
  `/api/character-file/${seg(handle)}/${seg(file)}`

/**
 * Stitch this scene's takes into one file.
 *
 * The takes travel as `{job_id, file}` and never as bytes: they are already on
 * the volume, so sending them back up would be uploading a scene to the machine
 * that made it. Answers with the job id to poll, like every other queued job —
 * the export is not a special case, it is CPU work with a status record.
 */
export const exportScene = (takes: { job_id: string; file: string }[]) =>
  post<Record<string, unknown>>('/api/export-scene', { takes })

export const gallery = (before = 0, limit = 200) =>
  api<Record<string, unknown>>(`/api/gallery?before=${before}&limit=${limit}`)
export const fileUrl = (jobId: string, name: string) =>
  `/api/file/${seg(jobId)}/${seg(name)}`
/**
 * The same result at 320px, for anywhere it is shown small.
 *
 * Beside `fileUrl` because the pair is the decision, and separating them is
 * how one gets used where the other belongs — which is what happened: the grid
 * served full-resolution originals into 232px cells. The cost that mattered was
 * not the bytes but the descriptor `/api/file` holds open for the length of the
 * transfer, which is what refuses `volume.reload()` and freezes the listing.
 */
export const coverUrl = (jobId: string, name: string) =>
  `/api/cover/${seg(jobId)}/${seg(name)}`
export const deleteOutput = (jobId: string, body?: unknown) =>
  post<Record<string, unknown>>(`/api/outputs/${seg(jobId)}/delete`, body)
export const purgeOutputs = (body?: unknown) =>
  post<Record<string, unknown>>('/api/outputs/purge', body)

/* ---- upload ---------------------------------------------------------- */

/**
 * The one route that is not JSON. It streams multipart, so it takes a FormData
 * and must not have a Content-Type set by hand — the browser has to write the
 * boundary.
 */
export async function upload(form: FormData): Promise<Res<Record<string, unknown>>> {
  return api<Record<string, unknown>>('/api/upload', { method: 'POST', body: form })
}
