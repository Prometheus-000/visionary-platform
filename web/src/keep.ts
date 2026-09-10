import { refreshArsenal } from './scene/arsenal'
import { SESSION } from './datasets/session'
import { newMember, newShot, seedShotIds } from './scene/model'
import { newRegion, seedRegionIds, useStore, type Store } from './store'
import type { PoolFile } from './scene/model'

/**
 * What survives a reload, and nothing further.
 *
 * **A reload is not a new session, and losing the scene to one was never a
 * policy.** "It never remembers unless told" governs what comes back *tomorrow*
 * — the Arsenal is the telling, and a character you did not save is still gone
 * when the tab closes. It never meant that ⌘R destroys the cast, the pool, the
 * timeline, the dialogue and the sources with no warning and nothing to recover,
 * which is what it did: the store was plain memory, so one stray reload cost
 * everything except the characters somebody had thought to press Save on.
 *
 * The lifetime this wants is the one `datasets/session.ts` and `gallery/mine.ts`
 * already name in the same words — *survives a reload and dies with the tab,
 * which is exactly the lifetime an unsaved thing should have.* This is that rule
 * applied to the composer. Nothing is remembered across sessions; a new tab
 * opens empty, because `SESSION` lives in `sessionStorage` and a new tab mints a
 * new one, so the record written by the old tab is addressed by an id that no
 * longer exists anywhere.
 *
 * **IndexedDB rather than `sessionStorage`, and only because of the bytes.**
 * `PoolFile.b64` is what a run uploads — nine 1536px references is H3's own
 * maximum and several megabytes, and a reference video is sent whole — so the
 * ~5 MB `sessionStorage` quota is not a margin, it is the common case. IDB has
 * no such ceiling. What IDB does not have is `sessionStorage`'s lifetime, so
 * that is supplied here: records are keyed by `SESSION` and swept by age.
 *
 * **Two records, split on where the bytes are.** Typing writes the main record
 * every half second; the pool, the keyframes and the region photographs are
 * written only when one of them actually changes. Without the split every
 * keystroke re-serialised however many megabytes of base64 the session had
 * accumulated, which is a fix that costs more than the thing it fixes — the
 * same arithmetic `shrinkB64` does about PNG.
 */

const DB = 'visionary-keep'
const TABLE = 'sessions'

/**
 * How long a record outlives the tab that wrote it.
 *
 * Not zero, and it cannot be: there is no unload event a browser will let you
 * finish IDB work in, and sweeping every record that is not this tab's would
 * delete a *second* live tab's scene the moment this one loads. So the sweep is
 * by age, and a dead tab's record is unreachable long before it is collected —
 * its key is a `SESSION` id that died in that tab's `sessionStorage` and exists
 * nowhere else. Twelve hours is "not the same working day".
 */
const HORIZON = 12 * 60 * 60 * 1000

/** Written on every record and checked on read. Derived from the field lists
 *  below rather than hand-numbered, so adding or dropping a persisted field
 *  invalidates yesterday's records for free — nobody remembers to bump a
 *  version, and a build that changed the shape would otherwise restore
 *  `undefined` into a field the page renders. */
const shapeOf = () => [...MAIN, ...BIN].join(',')

/**
 * Everything authored that carries no base64. Cheap to write, so it goes on any
 * change.
 *
 * `mode` is deliberately absent: it is where you *are*, not what you made, and
 * coming back into the Sheet room in front of a canvas whose six wells emptied
 * with the reload is worse than landing on Generate. `kind` is here because the
 * side you were composing on is not a room, it is which half of the work exists.
 */
const MAIN = [
  'kind', 'prompt', 'negative', 'negOn', 'shot', 'loras', 'stash',
  'img', 'vid', 'gpu', 'container', 'scene', 'shotSel', 'takes', 'continueFrom',
  // The taken-over document and its pane together. `doc` alone would restore a
  // run driven by an override with nothing on screen saying so — the one bit
  // `SourcePane` exists to keep visible.
  'doc', 'docOpen',
  'regionWeight', 'styleStrength',
] as const satisfies readonly (keyof Store)[]

/**
 * The fields that can hold base64, plus the small ones that are positional with
 * them.
 *
 * `refRoles` is index-parallel to `refs` and `rsel` indexes `regions`, so they
 * ride in the same record as the array they point into: two records are two
 * writes, and a torn pair is a role on the wrong picture — the invariant the
 * store's own comment says has to hold or every face lands in the wrong
 * rectangle. `autoFirst` travels with `keyframe` for the same reason: it says
 * who put that frame there, and a warning that names the wrong actor is a
 * warning about a decision nobody made.
 */
const BIN = [
  'pool', 'keyframe', 'autoFirst', 'refs', 'refRoles', 'refVids',
  'regions', 'rsel', 'frame',
  // The playground rides here, not MAIN, because its attachments are base64.
  // The graph inside it is what you made — hours of rewiring — and losing it
  // to a reload would be the exact failure this file exists to prevent.
  'pg',
] as const satisfies readonly (keyof Store)[]

type Record_ = { session: string; touched: number; shape: string; data: Partial<Store> }

// ── the store of record ─────────────────────────────────────────────────────

/** Null once the browser has told us it has no IndexedDB — private mode, or a
 *  policy. Everything below degrades to a no-op and the app is exactly the app
 *  it was before this file existed. */
let handle: Promise<IDBDatabase> | null = null
let off = false

function open(): Promise<IDBDatabase> | null {
  if (off) return null
  if (typeof indexedDB === 'undefined') { off = true; return null }
  handle ??= new Promise<IDBDatabase>((res, rej) => {
    const r = indexedDB.open(DB, 1)
    r.onupgradeneeded = () => { r.result.createObjectStore(TABLE) }
    r.onsuccess = () => { res(r.result) }
    r.onerror = () => { rej(r.error ?? new Error('indexedDB.open failed')) }
    // Another tab is holding an old version open. Nothing here is worth
    // blocking a page load on, so it fails and the session runs unkept.
    r.onblocked = () => { rej(new Error('blocked')) }
  }).catch((e: unknown) => { off = true; throw e })
  return handle
}

async function write(key: string, rec: Record_): Promise<void> {
  const db = await open()
  if (!db) return
  await new Promise<void>((res, rej) => {
    const tx = db.transaction(TABLE, 'readwrite')
    tx.objectStore(TABLE).put(rec, key)
    tx.oncomplete = () => { res() }
    tx.onerror = () => { rej(tx.error ?? new Error('put failed')) }
    tx.onabort = () => { rej(tx.error ?? new Error('put aborted')) }
  })
}

async function read(key: string): Promise<Record_ | null> {
  const db = await open()
  if (!db) return null
  return new Promise<Record_ | null>((res, rej) => {
    const r = db.transaction(TABLE, 'readonly').objectStore(TABLE).get(key)
    r.onsuccess = () => { res((r.result as Record_ | undefined) ?? null) }
    r.onerror = () => { rej(r.error ?? new Error('get failed')) }
  })
}

/** Records nothing can reach any more. See `HORIZON` for why it is an age and
 *  not "every session but mine". */
async function sweep(): Promise<void> {
  const db = await open()
  if (!db) return
  await new Promise<void>((res) => {
    const tx = db.transaction(TABLE, 'readwrite')
    const cur = tx.objectStore(TABLE).openCursor()
    const cutoff = Date.now() - HORIZON
    cur.onsuccess = () => {
      const c = cur.result
      if (!c) return
      const rec = c.value as Record_ | undefined
      // A record with no timestamp is from a build that did not write one, which
      // makes it exactly as unreadable as an expired one.
      if (!rec?.touched || rec.touched < cutoff) c.delete()
      c.continue()
    }
    tx.oncomplete = () => { res() }
    // A sweep that failed is a little disk held, not a broken session.
    tx.onerror = () => { res() }
    tx.onabort = () => { res() }
  })
}

// ── the pool's object URLs ──────────────────────────────────────────────────

/**
 * `PoolFile.url` is an object URL, and an object URL is a handle on a blob in
 * *this* document. It does not survive a reload and storing it would restore a
 * thumbnail pointing at nothing, so it is dropped on the way out and rebuilt
 * from the bytes on the way back.
 *
 * The blob is created with no declared type on purpose. `url` is read in exactly
 * one place — `Material`'s `<img src>`, images only; a clip and a recording show
 * their channel as a glyph — and an `<img>` sniffs the bytes it is given. The
 * alternative is carrying a MIME through the pool that `shrinkB64` can already
 * have invalidated by re-encoding a PNG to JPEG, which is a field that would be
 * wrong in the one case it exists for.
 */
const stripUrls = (pool: Record<string, PoolFile>): Record<string, PoolFile> =>
  Object.fromEntries(Object.entries(pool).map(([k, f]) => [k, { ...f, url: '' }]))

async function rebuildUrls(
  pool: Record<string, PoolFile>,
): Promise<Record<string, PoolFile>> {
  const out: Record<string, PoolFile> = {}
  for (const [k, f] of Object.entries(pool)) {
    if (!f?.b64 || !f.kind) continue
    try {
      // `fetch` on a data URL rather than `atob` plus a loop: the decode is
      // native and one reference video is tens of megabytes of base64, which is
      // a visible pause to walk a character at a time.
      const blob = await (await fetch(`data:;base64,${f.b64}`)).blob()
      out[k] = { ...f, url: URL.createObjectURL(blob) }
    } catch {
      // A file that would not decode is left out of the pool, which is the state
      // every reader already handles: `Material` filters refs whose file is
      // missing and `readScene` drops them from the payload, so the label
      // numbering stays correct rather than pointing one picture along.
    }
  }
  return out
}

/**
 * Fill in fields a record predates.
 *
 * **The shape stamp catches a changed field *list*, and this catches the case it
 * cannot see: a field added inside `Region`, `Shot` or `CastMember`.** `regions`
 * was in the list before `Region.name` existed and is in it after, so a record
 * written by yesterday's build passes the stamp and restores boxes with no
 * `name` — and `handleOf(undefined)` throws inside a render, which takes the
 * whole card down. The window for that is a deploy: somebody reloads a tab whose
 * record was written by the build before.
 *
 * Each row is merged over the constructor that makes a fresh one, so a missing
 * field takes the default the constructor would have given it and everything
 * stored wins over it. `mine.ts`'s rule, applied to structures rather than to a
 * list: this is parsed from storage that survives a reload, so it is checked
 * rather than trusted.
 */
function normalize(patch: Partial<Store>): Partial<Store> {
  if (patch.regions) {
    patch.regions = patch.regions.map((x) => ({ ...newRegion(), ...x }))
  }
  if (patch.scene) {
    patch.scene = {
      ...patch.scene,
      shots: (patch.scene.shots ?? []).map((x) => {
        const base = newShot()
        return { ...base, ...x, say: { ...base.say, ...x?.say } }
      }),
      cast: (patch.scene.cast ?? []).map((x) => ({ ...newMember('subject'), ...x })),
      // A tab that stored its scene before sources existed hands back a scene
      // without the key, and `live()` reads `Object.values(sc.sources)` on the
      // way to first paint — an undefined here is the restore taking the whole
      // app down over a field the person never used.
      sources: patch.scene.sources ?? {},
    }
  }
  return patch
}

// ── restore ─────────────────────────────────────────────────────────────────

/**
 * Put back what this tab had, before first paint.
 *
 * Awaited in `main.tsx` beside `applyTheme`, for the same reason that one is:
 * the alternative is the empty composer painting first and the scene arriving
 * over it a frame later, which reads as the app losing your work and then
 * changing its mind. Failures are silent and total — an unreadable record is the
 * session you already had, an empty one.
 */
export async function restore(): Promise<void> {
  try {
    void sweep()
    const [main, bin] = await Promise.all([
      read(`${SESSION}#main`), read(`${SESSION}#bin`),
    ])
    const shape = shapeOf()
    const patch: Partial<Store> = {}
    if (main?.shape === shape) Object.assign(patch, main.data)
    if (bin?.shape === shape) Object.assign(patch, bin.data)
    if (!Object.keys(patch).length) return
    normalize(patch)
    if (patch.pool) patch.pool = await rebuildUrls(patch.pool)
    // **Past every id that came back, or a fresh row collides with a restored
    // one.** Both counters are module-level and restart at zero on a reload, so
    // the first shot added after a restore was `s1` again — a second row with a
    // live row's id, which `patchShot` writes to both of and React keys as one.
    seedShotIds([
      ...(patch.scene?.shots ?? []).map((x) => x.id),
      ...(patch.scene?.cast ?? []).map((x) => x.id),
    ])
    seedRegionIds((patch.regions ?? []).map((x) => x.id))
    useStore.setState(patch)
    // **A restored cast re-reads the shelf.** This is the reason `PoolFile.from`
    // exists: persistence is what lets a recalled character outlive the library
    // it was copied from, so the moment that becomes possible is the moment the
    // second half of *edits in the library propagate* has to run. Not awaited —
    // it is a fetch per recalled file and the composer must not wait on the
    // shelf to paint; what it finds arrives a beat later, in place.
    void refreshArsenal()
  } catch {
    // Storage is a courtesy. There is no error worth showing on a cold load for
    // a feature whose failure mode is the behaviour that shipped for a year.
  }
}

// ── keeping ─────────────────────────────────────────────────────────────────

/** Long enough that a sentence is one write rather than thirty, short enough
 *  that ⌘R a beat after the last word still finds it. */
const SETTLE = 500

const slice = (st: Store, keys: readonly (keyof Store)[]): Partial<Store> =>
  Object.fromEntries(keys.map((k) => [k, st[k]])) as Partial<Store>

/**
 * Follow the store, and write what changed.
 *
 * Dirty by identity rather than by value: every writer in the store returns new
 * objects for what it touched, so a reference comparison across the whole field
 * list is both exact and free — and it is what keeps a hover, a menu open or a
 * poll's answer from costing a disk write.
 */
export function keep(): () => void {
  let lastMain: unknown[] | null = null
  let lastBin: unknown[] | null = null
  let timer = 0

  const flush = () => {
    window.clearTimeout(timer)
    timer = 0
    const st = useStore.getState()
    const now = Date.now()
    const shape = shapeOf()
    const mainVals = MAIN.map((k) => st[k])
    const binVals = BIN.map((k) => st[k])
    if (!lastMain || MAIN.some((_, i) => mainVals[i] !== lastMain![i])) {
      lastMain = mainVals
      void write(`${SESSION}#main`, {
        session: SESSION, touched: now, shape, data: slice(st, MAIN),
      }).catch(() => undefined)
    }
    if (!lastBin || BIN.some((_, i) => binVals[i] !== lastBin![i])) {
      lastBin = binVals
      const data = slice(st, BIN)
      data.pool = stripUrls(st.pool)
      void write(`${SESSION}#bin`, {
        session: SESSION, touched: now, shape, data,
      }).catch(() => undefined)
    }
  }

  const unsub = useStore.subscribe(() => {
    if (timer) return
    timer = window.setTimeout(flush, SETTLE)
  })

  // **The half second the debounce is holding is the half second a reload takes
  // away.** `visibilitychange` is the last event a browser reliably delivers
  // before a navigation, and an IDB transaction opened inside it is allowed to
  // finish — `beforeunload` is not, and `unload` is not fired at all on mobile.
  // Nothing here can be awaited, so this starts the write and lets the browser
  // land it.
  const hide = () => { if (document.hidden && timer) flush() }
  document.addEventListener('visibilitychange', hide)
  window.addEventListener('pagehide', flush)

  return () => {
    unsub()
    document.removeEventListener('visibilitychange', hide)
    window.removeEventListener('pagehide', flush)
    window.clearTimeout(timer)
  }
}

/**
 * Forget this tab's record.
 *
 * For the one gesture that means it: starting over. Nothing calls it yet, and it
 * exists so that when something does, the answer is one function rather than a
 * second spelling of the key format.
 */
export async function forget(): Promise<void> {
  try {
    const db = await open()
    if (!db) return
    const tx = db.transaction(TABLE, 'readwrite')
    tx.objectStore(TABLE).delete(`${SESSION}#main`)
    tx.objectStore(TABLE).delete(`${SESSION}#bin`)
  } catch { /* see `restore` */ }
}
