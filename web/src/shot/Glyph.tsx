/**
 * One drawing, eighty-seven tiles.
 *
 * Every tile is this same skeleton; the class on the `<svg>` is what turns it into
 * a push-in or a candle. That is not a saving of bytes — it is the reason the
 * animations are possible at all: the stylesheet animates named parts of one known
 * shape, so a dolly-out is `.gl-ca-pull .tk{animation:gCreep…}` rather than eighty
 * hand-drawn sequences. See the `.gl-*` block in ui.css.
 *
 * The posts are spaced exactly 20 so a 20px stream loops with no seam, and they
 * are what makes lateral movement visible at all: a horizontal line translated
 * horizontally is a correct animation of nothing.
 *
 * This is also the one place on the page where an icon can *teach*, which is why
 * the palette is tiles rather than a menu of words: a tile shows a dolly-out,
 * which is the thing neither the word nor a static picture does.
 */
const TICKS = [-34, -14, 6, 26, 46, 66]
const GRAIN = [[7, 6], [16, 11], [27, 6], [35, 14], [10, 20], [21, 25], [38, 22], [30, 17], [5, 13]]

export function Glyph({ cls }: { cls?: string }) {
  return (
    <svg className={`gl gl-${cls ?? ''}`} viewBox="0 0 44 30" aria-hidden="true">
      <path className="lt" d="M4 -2H20L11 32H-8Z" />
      <g className="wo">
        <line className="hz" x1="-40" y1="21" x2="84" y2="21" />
        <g className="tk">
          {TICKS.map((x) => <path key={x} d={`M${x} 21v-4.5`} />)}
        </g>
        <ellipse className="s2" cx="22" cy="16" rx="3.2" ry="4.8" />
        <ellipse className="s1" cx="22" cy="16" rx="3.2" ry="4.8" />
        <circle className="ob" cx="22" cy="17.5" r="1.7" />
      </g>
      <path className="pv" d="M3 30 13 21M41 30 31 21" />
      <path className="bu" d="M26 5h13a2 2 0 0 1 2 2v5a2 2 0 0 1-2 2h-7l-4 3v-3a2 2 0 0 1-2-2V7a2 2 0 0 1 2-2z" />
      <path className="tx" d="M12 22h20M17 26h10" />
      <g className="gr">
        {GRAIN.map(([cx, cy]) => <circle key={`${cx}:${cy}`} cx={cx} cy={cy} r=".75" />)}
      </g>
      <g className="bars">
        <rect x="-8" y="-1" width="60" height="4.6" />
        <rect x="-8" y="26.4" width="60" height="4.6" />
      </g>
      <rect className="fr" x=".5" y=".5" width="43" height="29" rx="2.5" />
    </svg>
  )
}
