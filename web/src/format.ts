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
