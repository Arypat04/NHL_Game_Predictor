# Design — Line Lab

<!-- impeccable:design-doc 1 -->

The world is a **precision laboratory instrument**. Line Lab reads out model win
probabilities and plots each game's edge as a measured deviation from the market
baseline. Not a sportsbook, not a tipsheet, not a generic dark dashboard. The
world lends four things only — **type, palette, density, and one signature move**
— while navigation and controls stay standard web components.

Source of truth: `frontend/src/index.css` (tokens + base) and
`frontend/src/App.css` (shell, controls, surfaces, states).
`ProbScale` is the signature. Surface brief: `.impeccable/surfaces/frontend-src-app-jsx.md`.

## Signature move — the edge as a measured deviation

`frontend/src/components/ProbScale.jsx`. A calibrated probability axis read like
an instrument: left = away certain (0%), right = home certain (100%), center =
even. A **hollow diamond** marks the market's implied line; a **filled needle**
is the model's reading; the **caliper** between them is the edge — cyan when it
favors the best bet (value), amber when that bet runs against the market. On mount
the needle settles from the reference to its value (one authored moment, exponential
ease-out `cubic-bezier(0.16,1,0.3,1)`, ~0.9s; reduced-motion jumps to final). When
live odds are present the scale carries the market diamond and caliper; when odds
are out of quota it shows the model reading alone against an even baseline.

## Type

Self-hosted (no hosted font stylesheet), `frontend/src/assets/fonts/`.

- **Archivo** (500/600/700/800) — UI, headings, team codes, labels. `--font-ui`.
- **Spline Sans Mono** (400/500/600/700) — every figure: probabilities, odds,
  scores, %, ticks. `--font-mono`. Mono is functional (tabular alignment +
  instrument readout), never decorative. Always `font-variant-numeric:
  tabular-nums slashed-zero`.

Display wordmark ~22px/800, tracking -0.03em. Section titles are mono 12px,
letter-spacing 0.14em, uppercase, with a rotated-square tick — no eyebrow/kicker.

## Palette

Committed, not grey-plus-one-accent. Two themes; dark is default (the
night-of-games use scene). Full token list in `index.css`.

| Role | Dark | Light |
|---|---|---|
| ground | `#0a0e16` | `#eceff4` |
| panel | `#111825` | `#ffffff` |
| graph grid (minor/major) | cyan @ 4.5% / 7.5% | blue @ 5% / 9% |
| ink / dim / faint | `#e9eef6` / `#9fb0c6` / `#6d7f98` | `#0e1622` / `#47566b` / `#68788f` |
| **signal** (for / live / brand) | `#37d2c6` | `#0b9a90` |
| **counter** (against / caution) | `#f2a73c` | `#b9700a` |
| market reference | `#7f8ca3` | `#64748b` |
| crit (true errors only) | `#ef5f5f` | `#c0392b` |

**Colorblind-safe by rule:** data meaning uses **cyan vs amber**, never red/green,
and never color alone — always paired with sign (+/−), direction (▲/▼ icons), or
a drawn shape (HIT check / MISS cross). `crit` red is reserved for genuine errors.

## Structure & density

- Shell: sticky instrument header (wordmark + status strip + theme toggle) over
  the graph-paper ground. `--maxw: 1160px`.
- Controls (standard web): segmented sport + bet-type (`.seg`), horizontal date
  strip (`.datestrip`), a sort toggle (value / confidence / time), a
  min-confidence range (`.filter`), and a "splits only" toggle.
- One unified board (`frontend/src/components/BoardTable.jsx`) — a single
  **reading list** (`.readings` / `.reading`), one row per game, not two disjoint
  lists of the same games. Each row carries the model **Pick** and the best-value
  **Bet**; when they disagree the row shows a `SPLIT` chip and spells out bet vs
  pick, otherwise it collapses to one line. A 4-column grid
  `time · matchup · scale · readout`: dense rows, hairline rules, hover highlight.
  On ≤720px rows collapse to `matchup + readout` over a full-width scale; the
  status strip folds to a compact strip below the header.
- Geometry: small radii (3–8px). Shadows always carry offset **and** blur
  (`--shadow`); no zero-blur halos, no hard offset shadows.

## States (degrade with dignity)

`frontend/src/components/States.jsx`, chosen from the API error's `kind`
(`services/api.js`): **warming** (503, pulsing instrument), **degraded** (401 =
odds free-tier quota — amber "feed unavailable"), **offline** and **error** (both
with a Retry action), **filtered-empty** (games exist but the confidence filter
or splits-only toggle hid them — never "no games"), **empty** ("no games on the
slate"), and shimmer **loading skeletons** shaped like readings. The warming
state is honest-by-construction: the app polls while the server is cold and the
readings appear on their own. Every control has hover / active / disabled /
focus-visible.

## Browser surfaces (themed, not default)

Selection, caret, `accent-color`, focus-visible ring, and custom scrollbars all
draw from the palette (`index.css`). Tabular numerals everywhere figures appear.

## Icons

Authored SVG set, one 1.5px stroke, 24px grid (`components/Icons.jsx`). No emoji,
no unicode glyphs standing in for icons.

## This world refuses

Hero-metric stat cards, same-size icon+heading cards, eyebrows/kickers, gradient
text, glass-as-decoration, colored left-borders, hard offset shadows, red/green
outcome color, emoji icons, hosted font stylesheets, monospace-as-costume. The
graph-paper background is a deliberate, committed armature of the instrument
world (the detector's advisory is overridden by the pinned world, which its own
rule exempts for measurement surfaces).

## Extending (NBA / NFL / player props)

Sports are not hardcoded to two — add to `SportTabs`. Player Props has a
disabled affordance already. New surfaces inherit this world: reading lists, the
ProbScale (or a mode-appropriate instrument readout), mono figures, cyan/amber
signals, the same states. Keep the world's four contributions; don't reach for
new ornament.
