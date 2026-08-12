/**
 * The app state, in one store.
 *
 * The vanilla page kept 27 module-level `let`s. They are the state, and they
 * map near one-to-one onto what is here — the port is a rehoming, not a
 * redesign, because every one of them exists for a reason recorded elsewhere.
 *
 * **What deliberately did not come across.** Four of the 27 are not state at
 * all, they are the residue of doing this by hand, and putting them in a store
 * would make the port slower than the page it replaces:
 *
 *   growing      a reentrancy latch inside `autoGrow`
 *   dragEnd      a pointer-gesture bookkeeping value
 *   menuEl       the open popover's element
 *   promptCaret  where the caret was when `+ LoRA` was pressed
 *
 * Those become refs. The rule the split follows: **if a value changes at
 * pointer rate, or is a handle on a DOM node, it is a ref.** A region drag
 * writes transforms directly and commits to the store on `pointerup` only —
 * state-per-frame would make the React canvas slower than the vanilla one it
 * replaces, which is the one way this port can be a straight downgrade.
 */
import { create } from 'zustand'
import type { AppState, ShotPill, VideoModel } from './api/types'

export type Kind = 'image' | 'video'
export type Mode = 'generate' | 'train' | 'datasets'

/** A region is a rectangle in 0..1 of the frame, plus what it is told to be.
 *  Its prompt takes the same `<lora:…>` syntax as the main field, which is the
 *  whole reason a region has no LoRA select of its own. */
export type Region = {
  id: string
  x: number
  y: number
  w: number
  h: number
  prompt: string
  /** Carried as a bool in the job record, never the bytes — it is polled. */
  ref?: string | null
}

export type Store = {
  /* ---- served vocabulary. The server is the authority; never restate it. -- */
  state: AppState | null
  stateError: string | null
  setState: (s: AppState) => void
  setStateError: (e: string | null) => void

  /* ---- where you are ---------------------------------------------------- */
  mode: Mode
  kind: Kind
  setMode: (m: Mode) => void
  /** The prompt survives the switch, because a shot you described as a still
   *  is the same sentence you would describe as a clip. */
  setKind: (k: Kind) => void

  /* ---- the composer ----------------------------------------------------- */
  prompt: string
  negative: string
  shot: ShotPill[]
  setPrompt: (p: string) => void
  setNegative: (n: string) => void
  toggleShot: (key: string) => void
  setShot: (pills: ShotPill[]) => void

  /* ---- regions ---------------------------------------------------------- */
  regional: boolean
  regions: Region[]
  selected: string | null
  setRegional: (on: boolean) => void
  setRegions: (r: Region[]) => void
  select: (id: string | null) => void

  /* ---- jobs ------------------------------------------------------------- */
  job: string | null
  setJob: (id: string | null) => void
}

export const useStore = create<Store>((set) => ({
  state: null,
  stateError: null,
  setState: (s) => set({ state: s, stateError: null }),
  setStateError: (e) => set({ stateError: e }),

  mode: 'generate',
  kind: 'image',
  setMode: (mode) => set({ mode }),
  setKind: (kind) => set({ kind }),

  prompt: '',
  negative: '',
  shot: [],
  setPrompt: (prompt) => set({ prompt }),
  setNegative: (negative) => set({ negative }),
  setShot: (shot) => set({ shot }),
  toggleShot: (key) =>
    set((s) => ({
      shot: s.shot.some((p) => p.key === key)
        ? s.shot.filter((p) => p.key !== key)
        : [...s.shot, { key }],
    })),

  regional: false,
  regions: [],
  selected: null,
  setRegional: (regional) => set({ regional }),
  setRegions: (regions) => set({ regions }),
  select: (selected) => set({ selected }),

  job: null,
  setJob: (job) => set({ job }),
}))

/** The chosen video family, or null on the image side. The composer shows only
 *  the controls this says the model reads — a control that is present but
 *  ignored is worse than one that is absent. */
export function videoModel(s: Store, key: string): VideoModel | null {
  if (s.kind !== 'video' || !s.state) return null
  return s.state.video_models.find((m) => m.key === key) ?? null
}
