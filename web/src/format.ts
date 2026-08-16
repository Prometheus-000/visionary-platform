/**
 * Sizes, spelled the way the catalogue spells them.
 *
 * Ported exactly rather than rewritten, and the rewrite is what makes the note
 * worth having: a plausible-looking 1024-based formatter with one decimal
 * rendered the same eight LoRAs as "5.4 GB" where the page says "5.85 GB". Both
 * are defensible in isolation; side by side one of them is wrong, and the
 * parity check only caught it because the two totals were printed next to each
 * other — it had asserted the total was *present*, not what it said.
 *
 * Decimal GB, because the catalogue's own `approx_gb` and `size_gb` are decimal
 * and a LoRA measured one way next to a checkpoint measured the other is a
 * comparison nobody can make. Two decimals to match what those are rounded to.
 * MB below a gigabyte, floored at 1 so a small file is never "0 MB".
 */
export const fmtBytes = (b: number): string => {
  const n = Number(b) || 0
  return n >= 1e9 ? `${(n / 1e9).toFixed(2)} GB` : `${Math.max(1, Math.round(n / 1e6))} MB`
}

/**
 * One file on the volume, at the precision a decision needs.
 *
 * Deliberately not `fmtBytes`. That one spells the *catalogue*: decimal GB to
 * two places, floored at 1 MB, because what it labels is a 17 GB checkpoint. A
 * dataset image is 300 KB to 12 MB, and floored-MB renders an 800 KB JPEG and a
 * 1.4 MB PNG as "1 MB" each — which is exactly the comparison the duplicate
 * review exists to make. The tile badge and the review share this one so that
 * the same file is never two sizes on one screen.
 */
export const fmtFileSize = (b: number): string => {
  const n = Number(b) || 0
  return n >= 1048576 ? `${(n / 1048576).toFixed(1)} MB` : `${Math.round(n / 1024)} KB`
}
