import { characterFileUrl, characters, saveCharacter } from '../api/routes'
import { failed } from '../api/client'
import { useStore } from '../store'
import { intake } from './pool'
import type { CastMember } from './model'

/**
 * The Arsenal's first shelf: characters, saved deliberately, recalled by name.
 *
 * Saving is a choice and never a side effect — "it never remembers unless
 * told" — and recall is the gesture that already exists: typing `@ma…` offers
 * `maya — saved` beside "New subject", so the library never grows a browser
 * panel. The drawer rule, kept: the name you are already typing is the recall.
 *
 * What travels is the pool's own bytes. `PoolFile.b64` is exactly what a run
 * uploads, so a save costs no re-encode — and a recall walks back through
 * `intake`, which is what keeps the content-keyed pool honest: a character
 * recalled twice, or recalled beside her own original photograph, still
 * resolves to one entry per picture.
 */

export type SavedRef = { file: string; kind: string; note?: string; sheet?: boolean }
export type SavedCharacter = {
  handle: string
  note: string
  retention: string
  refs: SavedRef[]
}

/** The saved cast, for the picker. A popover-speed listing — names, notes and
 *  filenames; never bytes. */
export async function listCharacters(): Promise<SavedCharacter[]> {
  const r = await characters()
  if (failed(r) || !Array.isArray((r as { characters?: unknown }).characters)) return []
  return (r as { characters: SavedCharacter[] }).characters
}

/** Save one member, whole. Resolves to an error sentence or null. */
export async function save(member: CastMember): Promise<string | null> {
  const pool = useStore.getState().pool
  const refs = member.refs.flatMap((r) => {
    const f = pool[r.fileId]
    if (!f) return []
    return [{ kind: f.kind, b64: f.b64,
              ...(r.note ? { note: r.note } : {}),
              ...(r.sheet ? { sheet: true } : {}) }]
  })
  const r = await saveCharacter(member.name, {
    note: member.note, retention: member.retention, refs,
  })
  return failed(r) ? r.error : null
}

/**
 * Rebuild a saved character into the live cast, files and all.
 *
 * The member is created synchronously by the caller — the picker needs a handle
 * to insert at the caret *now* — and this hydrates it: each file fetched off
 * its route, walked through `intake` so it lands in the pool exactly as a
 * dropped file would, then attached with its note and its sheet mark. Failures
 * are per-file and quiet: a character three of whose four files arrived is a
 * character, not an error, and the card shows exactly what made it.
 */
export async function hydrate(memberId: string, saved: SavedCharacter): Promise<void> {
  const st = useStore.getState()
  if (saved.note) st.patchCast(memberId, { note: saved.note })
  if (saved.retention) st.patchCast(memberId, { retention: saved.retention })
  for (const ref of saved.refs) {
    try {
      const res = await fetch(characterFileUrl(saved.handle, ref.file))
      if (!res.ok) continue
      const blob = await res.blob()
      const got = await intake(new File([blob], ref.file, { type: blob.type }))
      if (!got) continue
      const now = useStore.getState()
      now.addFile(got)
      now.attachSlot(memberId, got.id, got.kind)
      if (ref.note || ref.sheet) {
        now.patchRef(memberId, got.id, {
          ...(ref.note ? { note: ref.note } : {}),
          ...(ref.sheet ? { sheet: true } : {}),
        })
      }
    } catch {
      // A file that would not fetch is a gap on the card, visible there.
    }
  }
}
