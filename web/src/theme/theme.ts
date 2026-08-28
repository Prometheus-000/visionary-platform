/**
 * Colour, on the Strata model: monochrome by default, neon by choice.
 *
 * The rule this file exists to hold is Strata's, and the first implementation
 * broke it: **an arbitrary hue never touches the neutrals.** Surfaces and inks
 * are tinted only by `warmth`, on a bounded paper↔slate axis at chroma under
 * .025, which is what keeps a themed app from becoming "the brown app" or "the
 * purple app". The free hue belongs to the *accent*, where saturation is the
 * point and where a single filled action can carry it.
 *
 * Midnight is not a copy of the stylesheet's values — it IS the stylesheet.
 * Warmth 0 with chroma 0 removes every override, so the house theme cannot
 * drift from `ui.css`: there is nothing to keep in sync because there is no
 * second copy. Polar is its inverse ground.
 *
 * The phrase is the record. A mood typed into the panel compiles to these
 * numbers deterministically and every word's effect is itemized; the words are
 * what gets stored, the numbers are the receipt.
 *
 * Stored in localStorage rather than sessionStorage on purpose: a theme somebody
 * set by hand is `told`, in the sense the storage rule means it — it is the one
 * preference whose whole job is to still be there tomorrow.
 */

export type Pole = 'midnight' | 'polar'

export interface VsnTheme {
  pole: Pole
  /** Accent hue, 0–360. Free, because it never reaches the neutrals. */
  hue: number
  /** Accent chroma, 0–0.25. 0 is monochrome — the whole default. */
  chroma: number
  /** −1 slate … 0 neutral … 1 paper. The ONLY thing that tints surfaces. */
  warmth: number
  /** The prose that produced this, if any — kept verbatim. */
  phrase?: string
}

export const MIDNIGHT: VsnTheme = { pole: 'midnight', hue: 250, chroma: 0, warmth: 0 }
export const POLAR: VsnTheme = { pole: 'polar', hue: 250, chroma: 0, warmth: 0 }

/** Strata's own ceiling. .25 is neon; the accent is the one place it belongs. */
export const CHROMA_MAX = 0.25

const KEY = 'vsn-theme'

const clamp = (v: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, v))
const lerp = (a: number, b: number, t: number) => a + (b - a) * t

const ok = (l: number, c: number, h: number, a?: number) =>
  a === undefined
    ? `oklch(${l.toFixed(3)} ${c.toFixed(4)} ${h.toFixed(1)})`
    : `oklch(${l.toFixed(3)} ${c.toFixed(4)} ${h.toFixed(1)} / ${a})`

/** Every var the engine may write. Cleared as a set, so a stale override can't linger. */
const OWNED = [
  '--bg', '--fg', '--mut', '--dim', '--lift', '--lift-line', '--lift-cast',
  '--wash-1', '--wash-2', '--wash-3', '--sel', '--line', '--line-2',
  '--crit', '--crit-solid', '--crit-wash', '--crit-line', '--crit-fill', '--warn', '--ok',
  '--accent', '--accent-ink', '--accent-soft', '--accent-line', '--cast',
] as const

function tokens(t: VsnTheme): Record<string, string> {
  const hue = ((t.hue % 360) + 360) % 360
  const chroma = clamp(t.chroma, 0, CHROMA_MAX)
  const warmth = clamp(t.warmth, -1, 1)
  const dark = t.pole === 'midnight'

  /* Neutrals get a whisper of hue and nothing more: paper one way, slate the
     other. The two anchors do not interpolate through each other — a hue lerped
     from slate to paper passes through green, and a green-grey UI is not a
     thing anyone asked for. The sign picks the anchor; warmth sets how much.
     At warmth 0 the chroma is ui.css's own .008, so which anchor is chosen
     there is invisible — and at warmth 0 with no accent nothing is written. */
  const neutralHue = warmth >= 0 ? 85 : 250
  const neutralChroma = 0.008 + Math.abs(warmth) * 0.016

  /* The accent, Strata's formula verbatim, including the correction that makes
     it work on paper: a light ground needs a darker, slightly less saturated
     accent to hold contrast against the ink it carries. */
  const accentL = dark ? lerp(0.84, 0.78, chroma / CHROMA_MAX) : lerp(0.58, 0.52, chroma / CHROMA_MAX)
  const accentC = dark ? chroma : chroma * 0.87
  const accent = {
    '--accent': ok(accentL, accentC, hue),
    '--accent-ink': dark ? ok(0.16, Math.min(chroma * 0.35, 0.06), hue) : ok(0.98, 0.01, hue),
    '--accent-soft': ok(accentL, accentC, hue, dark ? 0.14 : 0.12),
    '--accent-line': ok(accentL, accentC, hue, dark ? 0.45 : 0.4),
    '--cast': castColor(t),
  }

  if (dark) {
    /* Every visible surface in ui.css is an alpha wash over the ground, so the
       wash is where a neutral cast has to live — but it stays a *neutral*: L is
       high and chroma is capped at .024, which is near the gamut ceiling at
       that lightness anyway. ui.css's own alphas are untouched; they carry the
       elevation order and only the colour is ours. */
    const washC = Math.min(0.024, neutralChroma * 1.5)
    const wash = (a: number) => ok(0.96, washC, neutralHue, a)
    return {
      ...accent,
      // The ground keeps its lightness. The page is black; warmth is a cast on
      // it, never a lift off it.
      '--bg': ok(0.069, neutralChroma, neutralHue),
      '--fg': ok(0.978, Math.min(0.01, neutralChroma * 0.5), neutralHue),
      '--mut': ok(0.649, 0.015 + Math.abs(warmth) * 0.01, neutralHue),
      '--dim': ok(0.626, 0.012 + Math.abs(warmth) * 0.01, neutralHue),
      '--lift': ok(0.227, neutralChroma * 1.2, neutralHue),
      '--lift-line': wash(0.13),
      '--wash-1': wash(0.03),
      '--wash-2': wash(0.05),
      '--wash-3': wash(0.07),
      // Selection is the one neutral that becomes the accent when there is one:
      // it is the app saying "this one", which is what an accent is for.
      '--sel': chroma > 0 ? accent['--accent-soft'] : wash(0.14),
      '--line': wash(0.1),
      '--line-2': wash(0.24),
    }
  }

  /* Polar: the inverse ground. The washes invert with it — black at the same
     alphas — and the status inks darken, because #f87171 is red ink calibrated
     for black and reads 2.5:1 on white, which is below every floor. */
  const wash = (a: number) => ok(0.2, Math.min(0.05, neutralChroma * 2), neutralHue, a)
  return {
    ...accent,
    '--bg': ok(0.985, neutralChroma * 0.8, neutralHue),
    '--fg': ok(0.09, Math.min(0.012, neutralChroma), neutralHue),
    '--mut': ok(0.42, 0.015 + Math.abs(warmth) * 0.01, neutralHue),
    '--dim': ok(0.45, 0.012 + Math.abs(warmth) * 0.01, neutralHue),
    '--lift': ok(1, neutralChroma * 0.4, neutralHue),
    '--lift-line': wash(0.14),
    '--lift-cast': '0 18px 48px rgba(0,0,0,.18)',
    '--wash-1': wash(0.035),
    '--wash-2': wash(0.055),
    '--wash-3': wash(0.075),
    '--sel': chroma > 0 ? accent['--accent-soft'] : wash(0.14),
    '--line': wash(0.12),
    '--line-2': wash(0.26),
    '--crit': '#dc2626',
    '--crit-solid': '#dc2626',
    '--crit-wash': 'rgba(220,38,38,.12)',
    '--crit-line': 'rgba(220,38,38,.3)',
    '--crit-fill': 'rgba(220,38,38,.08)',
    '--warn': '#b45309',
    '--ok': '#15803d',
  }
}

/**
 * What the mark in the header is painted with.
 *
 * The trigger *is* the state: monochrome it reads as punctuation after the
 * wordmark, and the moment there is an accent it carries the accent. With only
 * a warmth set it shows the neutral, pushed up in chroma so a cast you chose is
 * a cast you can see on a 7px dot.
 */
export function castColor(t: VsnTheme): string {
  if (t.chroma > 0) {
    const l = t.pole === 'midnight' ? lerp(0.84, 0.78, t.chroma / CHROMA_MAX) : 0.55
    return ok(l, t.pole === 'midnight' ? t.chroma : t.chroma * 0.87, t.hue)
  }
  if (t.warmth === 0) return 'var(--dim)'
  return ok(t.pole === 'midnight' ? 0.7 : 0.55, 0.04, t.warmth >= 0 ? 85 : 250)
}

export function applyTheme(t: VsnTheme) {
  const root = document.documentElement
  for (const p of OWNED) root.style.removeProperty(p)
  if (t.pole === 'midnight' && t.warmth === 0 && t.chroma === 0) {
    // The house theme is the stylesheet itself.
    delete root.dataset.vsnTheme
    delete root.dataset.vsnAccent
    return
  }
  root.dataset.vsnTheme = t.pole
  // Gates the accent patches. Without an accent the one filled action on a
  // screen stays white, which is ui.css's own answer and the monochrome default.
  if (t.chroma > 0) root.dataset.vsnAccent = 'on'
  else delete root.dataset.vsnAccent
  for (const [p, v] of Object.entries(tokens(t))) root.style.setProperty(p, v)
}

export function loadTheme(): VsnTheme {
  try {
    const raw = window.localStorage.getItem(KEY)
    if (!raw) return MIDNIGHT
    const t = JSON.parse(raw) as VsnTheme & { tint?: number }
    if ((t.pole === 'midnight' || t.pole === 'polar') && Number.isFinite(t.hue)) {
      // Themes stored before warmth and accent were separate carried both in
      // one `tint`; the closest honest reading of that is a warmth, since it is
      // what those themes actually looked like.
      const warmth = Number.isFinite(t.warmth) ? t.warmth : (t.tint ?? 0)
      return {
        pole: t.pole,
        hue: t.hue,
        chroma: clamp(Number.isFinite(t.chroma) ? t.chroma : 0, 0, CHROMA_MAX),
        warmth: clamp(warmth, -1, 1),
        phrase: t.phrase,
      }
    }
  } catch { /* blocked storage boots the house theme */ }
  return MIDNIGHT
}

export function saveTheme(t: VsnTheme) {
  try {
    window.localStorage.setItem(KEY, JSON.stringify(t))
  } catch { /* nothing to do — it simply won't survive the tab */ }
}

/* ---------------------------------------------------------------- */

export interface Receipt { word: string; effect: string }

const HUES: Record<string, number> = {
  crimson: 25, rust: 32, ember: 40, copper: 50, amber: 70, gold: 85, honey: 78,
  chartreuse: 108, lime: 115, moss: 125, meadow: 135, forest: 142, jade: 155,
  emerald: 160, mint: 165, teal: 178, cyan: 198, ice: 212, glacier: 220,
  ocean: 232, blue: 248, cobalt: 260, indigo: 275, violet: 295, ultraviolet: 302,
  orchid: 315, magenta: 330, rose: 350,
}

const MOODS: Record<string, Partial<VsnTheme>> = {
  // Chroma words move the accent. Warmth words move the neutrals. They are
  // separate because they are separate — `warm` is not a quieter `amber`.
  neon: { chroma: 0.24 }, electric: { chroma: 0.22 }, vivid: { chroma: 0.19 },
  rich: { chroma: 0.16 }, muted: { chroma: 0.07 }, dusty: { chroma: 0.05 },
  faded: { chroma: 0.04 }, pastel: { chroma: 0.08 },
  monochrome: { chroma: 0 }, greyscale: { chroma: 0 }, mono: { chroma: 0 },
  warm: { warmth: 0.55 }, paper: { warmth: 0.8 }, candle: { warmth: 0.7 },
  sepia: { warmth: 0.75 }, cool: { warmth: -0.55 }, slate: { warmth: -0.7 },
  arctic: { warmth: -0.9 }, clinical: { warmth: -0.6, chroma: 0 },
  night: { pole: 'midnight' }, midnight: { pole: 'midnight', chroma: 0, warmth: 0 },
  noir: { pole: 'midnight', chroma: 0, warmth: 0 }, dark: { pole: 'midnight' },
  polar: { pole: 'polar', chroma: 0, warmth: 0 }, day: { pole: 'polar' },
  light: { pole: 'polar' }, morning: { pole: 'polar' },
}

const STOP = new Set(['a', 'an', 'the', 'of', 'on', 'in', 'at', 'and', 'or', 'with', 'to', 'for'])

/** Deterministic: the same sentence always compiles to the same theme. */
export function compileMood(phrase: string, base: VsnTheme): { theme: VsnTheme; receipts: Receipt[] } {
  const words = phrase.toLowerCase().split(/[^a-z0-9]+/).filter(Boolean)
  const t: VsnTheme = { ...base, phrase: phrase.trim() }
  const receipts: Receipt[] = []
  let hueSet = false
  let chromaSet = false

  for (const w of words) {
    if (HUES[w] !== undefined) {
      t.hue = HUES[w]
      hueSet = true
      receipts.push({ word: w, effect: `hue ${HUES[w]}°` })
    } else if (MOODS[w]) {
      Object.assign(t, MOODS[w])
      if (MOODS[w].chroma !== undefined) chromaSet = true
      receipts.push({
        word: w,
        effect: Object.entries(MOODS[w]).map(([k, v]) => `${k} ${v}`).join(', '),
      })
    } else if (!STOP.has(w)) {
      receipts.push({ word: w, effect: 'silent' })
    }
  }

  // A colour word is a request for colour: naming a hue with the accent still
  // at zero would compile to the monochrome theme and read as "it did nothing".
  if (hueSet && !chromaSet && t.chroma === 0) t.chroma = 0.14
  return { theme: t, receipts }
}
