import { characterFileUrl, characters, saveCharacter } from '../api/routes'
import { failed } from '../api/client'
import { toB64 } from '../media/files'
import type { LoraChip } from '../lora/tokens'
import { attached, useStore, type Region } from '../store'
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
/** A pointer to a weight on the volume, never the weight. See `save_character`
 *  in app.py for why, and for what happens when the file is later deleted. */
export type SavedLora = { path: string; rel: string; strength: number }
export type SavedCharacter = {
  handle: string
  note: string
  retention: string
  refs: SavedRef[]
  lora?: SavedLora
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
 * The same shelf, reached from the image side.
 *
 * **A character placed on the canvas and a character cast in the composer are
 * the same person, and only one of them could be kept.** The video side has had
 * this since the Arsenal landed; the image side — the surface whose empty canvas
 * says *"double-click to place a character"* — had no save at all, so every
 * likeness built there died with the tab. Two ways to say who is in a shot is
 * already one more than the root file allows; two ways where only one remembers
 * is the version of that with a cost.
 *
 * What a box holds maps onto the record without inventing a field: its sentence
 * is the description, its identity photo is the one reference, and its LoRA is
 * the LoRA. What does *not* travel is the rectangle — where somebody stands is a
 * fact about this frame, not about them, and a saved character that dragged its
 * old coordinates into every future arrangement would be storing the wrong half.
 */
export async function saveRegion(r: Region, name: string): Promise<string | null> {
  const photo = attached(r, 'identity')
  const res = await saveCharacter(name, {
    note: r.prompt,
    // The image side has no retention control — that marker is H3 grammar, and
    // `_h3_label` is the only thing that reads it. Sent empty rather than
    // defaulted, so recalling into the composer takes *its* default rather than
    // one this side invented on somebody's behalf.
    retention: '',
    refs: photo ? [{ kind: 'image', b64: photo }] : [],
    ...(r.lora ? { lora: { path: r.lora.path, rel: r.lora.rel,
                           strength: r.lora.strength } } : {}),
  })
  return failed(res) ? res.error : null
}

/**
 * Rebuild a saved character into a box.
 *
 * The mirror of `hydrate`, and shorter for one reason: a region takes a single
 * likeness rather than a stack, so the first picture is the picture. The rest is
 * the same contract — per-file failures are quiet and partial, because a
 * character whose photograph would not fetch is still that character's sentence
 * and their weight.
 *
 * `index` is read fresh on every write rather than closed over: this awaits two
 * round trips, and selecting a different box inside them is a thing a hand does.
 */
export async function hydrateRegion(
  id: string,
  saved: SavedCharacter,
  index: { path: string }[],
): Promise<void> {
  const at = () => useStore.getState().regions.findIndex((x) => x.id === id)
  const here = at()
  if (here < 0) return
  useStore.getState().patchRegion(here, {
    name: saved.handle,
    // Their own sentence, and only into an empty field: a box you had already
    // written in is a box you meant, and overwriting it to recall a face would
    // be the recall deleting the thing the recall was for.
    ...(saved.note && !useStore.getState().regions[here]?.prompt.trim()
        ? { prompt: saved.note } : {}),
    // **Only if the weight is still there.** A LoRA deleted since the save is a
    // chip pointing at a path `/api/generate` would refuse, and a character that
    // recalls with three quarters of itself is the state this whole function is
    // built to produce — see the per-file rule above.
    ...(saved.lora && index.some((l) => l.path === saved.lora!.path)
        ? { lora: { path: saved.lora.path, rel: saved.lora.rel,
                    strength: saved.lora.strength,
                    textEncoder: null } as LoraChip }
        : {}),
  })
  const pic = saved.refs.find((x) => (x.kind || 'image') === 'image')
  if (!pic) return
  try {
    const res = await fetch(characterFileUrl(saved.handle, pic.file))
    if (!res.ok) return
    const b64 = await toB64(await res.blob())
    const now = at()
    if (b64 && now >= 0) useStore.getState().attach(now, 'identity', b64)
  } catch {
    // A photograph that would not fetch leaves the box with its sentence and
    // its weight, which is visible on the card and needs no sentence about it.
  }
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
