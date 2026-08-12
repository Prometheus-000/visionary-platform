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
import type { AppState, CompileResult, JobStatus, ShotPill } from './types'

const seg = encodeURIComponent

/* ---- state, settings, weights ---------------------------------------- */

export const getState = () => api<AppState>('/api/state')
export const getWhere = () => api<Record<string, unknown>>('/api/where')
export const setToken = (token: string) => post<{ ok?: boolean }>('/api/token', { token })
export const startDownload = (key: string, family?: string) =>
  post<{ job_id: string }>('/api/download', family ? { key, family } : { key })
export const startGdrive = (url: string, folder?: string) =>
  post<{ job_id: string }>('/api/gdrive', { url, folder })
export const downloadMissing = () => post<{ job_id: string }>('/api/download-missing')
export const deleteLoras = (paths: string[]) => post<{ ok?: boolean }>('/api/loras/delete', { paths })

/* ---- session --------------------------------------------------------- */

/** A draft belongs to the window that made it; this is the heartbeat that
 *  keeps it from being swept after fifteen minutes of silence. */
export const beat = (session: string) => post<{ ok?: boolean }>('/api/session', { session })

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
export const datasetInsight = (name: string) =>
  api<Record<string, unknown>>(`/api/datasets/${seg(name)}/insight`)
export const prependTrigger = (name: string, body: unknown) =>
  post<Record<string, unknown>>(`/api/datasets/${seg(name)}/prepend-trigger`, body)

/** Bytes off the volume, by their own route — never inlined into a polled
 *  record. A dict polled every two seconds must not grow with its result. */
export const thumbUrl = (name: string, file: string) =>
  `/api/thumb/${seg(name)}/${seg(file)}`
export const imageUrl = (name: string, file: string) =>
  `/api/image/${seg(name)}/${seg(file)}`

/* ---- jobs ------------------------------------------------------------ */

export const caption = (body: unknown) => post<{ job_id: string }>('/api/caption', body)
export const train = (body: unknown) => post<{ job_id: string }>('/api/train', body)
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
  references?: number
  ref_videos?: number
  ref_roles?: string[]
  first_frame?: boolean
  last_frame?: boolean
}) => post<CompileResult>('/api/compile', body)

/* ---- outputs and the gallery ----------------------------------------- */

export const gallery = () => api<Record<string, unknown>>('/api/gallery')
export const fileUrl = (jobId: string, name: string) =>
  `/api/file/${seg(jobId)}/${seg(name)}`
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
