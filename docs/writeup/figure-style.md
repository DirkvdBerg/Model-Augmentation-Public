# Figure Style

House style for every diagram in `docs/` and the thesis. Written down so figures stop
re-deriving it (v1 through v4 of the block scheme each drifted).

Source of the style: `docs/writeup/jan-blockscheme-v4.tex`, whose own header cites
**Hoekstra et al. (2025) Fig. 1**. Matching Jan's figures is the point. A figure that reads
as a sibling of his does half the communication before it says anything.

---

## 1. Build

| Item | Convention |
|---|---|
| Format | standalone TikZ, `\documentclass[border=12pt]{standalone}` |
| Packages | `tikz` (+ `arrows.meta`, `calc`), `amsmath`, `xcolor` only if greys are needed |
| Compile | `pdflatex <name>.tex` from `docs/` |
| Preview | `pdftoppm -png -r 150 <name>.pdf <name>` |
| Naming | `<topic>-v<N>.tex`; bump `N` for a structural revision, edit in place for a typo |
| Header comment | one line per version saying what changed and why (see v4 lines 1 to 7) |

Standalone crops to content, so the figure drops into the thesis at any `\includegraphics`
width without a bounding-box fight.

## 2. Ink

**Black on white. Greys carry meaning, colour does not.**

Permitted fills, and nothing else:

| Token | Value | Means |
|---|---|---|
| black | `black` | the object under discussion |
| mid grey | `black!55` | present but explicitly NOT part of the claim |
| light grey | `black!12` | a region or band (encoder window, horizon bar) |

Rationale: these figures print in black and white, get photocopied, and land in slide decks
with unknown colour management. Two regions separated on a time axis do not need hue to be
told apart. If a figure genuinely needs a third distinction, that figure is doing two jobs
and should be split.

Greying is a claim, not decoration: grey means "this exists in the model and is deliberately
not in the loss / not compared / has no path to the output". Never grey something merely to
de-emphasise it visually.

## 3. Node vocabulary

Reuse these names and sizes verbatim so blocks line up across figures.

```
block   draw, rounded corners, minimum width=26mm, minimum height=12mm, align=center
rblock  draw, rounded corners, minimum width=13mm, minimum height=11mm, align=center
small   draw, rounded corners, minimum width=15mm, minimum height=10mm, align=center
sum     draw, circle, minimum size=7mm, inner sep=0pt
jn      circle, fill, minimum size=3pt, inner sep=0pt        (signal junction dot)
```

Arrow tips are `>=Latex` globally. Signal buses are `thick`; concatenation bus-bars are
`line width=1.3pt` and carry `jn` dots at every tap, which is how a bundle is distinguished
from a summation.

## 4. Type

| Element | Size |
|---|---|
| Figure base font | `\footnotesize` |
| Block name | math symbol at base size, e.g. `$f_{\mathrm{base}}$` |
| Block gloss | `\scriptsize`, one line under the symbol |
| Signal labels, annotations | `\scriptsize` |
| Panel titles | `\footnotesize\bfseries` |

**Legibility floor.** `\scriptsize` in a standalone figure is 7 pt. Placed type must land at
**6 pt or larger**, so a figure may be scaled down to about `0.85x` of its native width and
no further. Decide the target placement width *before* drawing, because it fixes how much
content fits:

| Target | Placed width | Max native width | Verdict |
|---|---|---|---|
| IFAC/IEEE single column | 85 mm | 100 mm | fits a block diagram, not a multi-panel figure |
| Thesis page width | 150 mm | 175 mm | the working default for this project |

Both `jan-blockscheme-v4` (185 mm native) and `training-objective-v1` (173 mm native) are
**page-width figures**. Neither survives an 85 mm column, and neither should be squeezed into
one: reducing them to fit puts annotations at 3 to 4 pt.

Check before committing: `pdfinfo <name>.pdf` for the native width, then confirm
`placed / native >= 0.85`. If a figure needs to go in a narrow column, cut content or split
the panels. Do not shrink type.

## 5. Scale honesty

If a figure's argument rests on a ratio, **draw the ratio to scale**. Do not enlarge a small
element "so it can be seen": that discards the argument the figure exists to make.

The corollary, and it is not optional: **an element that is correct at 1 mm must carry its
message in an external label with a leader line**, never inside itself. A figure whose point
depends on the reader resolving a 1 mm feature is fragile; a figure whose point is stated in
a label anchored to that feature is not.

When two scales cannot share an axis, use stacked panels at true scale joined by a
magnification wedge. The wedge is structural, not decorative.

## 6. Text inside figures

- **No em-dashes**, per `CLAUDE.md`: not the Unicode character, not `---`, not `--` as
  punctuation. Use a colon, a comma, or a new clause.
  (TikZ path syntax `--` is unaffected: it is an operator, not punctuation.)
- Units in brackets: `[m]`, `[s]`, `[Hz]`.
- Numbers that carry the argument go in the label text, not only in the geometry
  (`1/120`, `n_f = 400`, `18 samples`).
- Prefer the symbol the code uses (`nf`, `na`, `stride`) next to the maths, so the figure
  and `gantry_dynamic/config.py` can be read against each other.

## 7. Legends and captions

**Prefer direct labelling; reach for a legend when direct labelling is not possible.** Label
an object where it sits whenever it can be reached. That removes a saccade and a memory
lookup, so the reader never has to hold "dashed = model" in their head while looking
somewhere else.

A legend is the right tool, not a failure, when:

- series overlap or are too dense to carry an inline label
- the same encoding recurs across panels (one key beats repeating it per panel)
- there are more than about four series, where inline labels become their own clutter
- the figure will be read out of context, lifted into a slide or scanned to directly, where a
  self-contained key is a service to the reader

`jan-blockscheme-v4` drops its legend because every object in it is labelled in place. That
is a judgement about that figure, not a prohibition.

Caption carries: what the figure shows, the parameter values it was drawn at, and whether
traces are real data or schematic. Never leave a reader guessing which.

## 8. Checklist before committing a figure

1. Compiles clean with `pdflatex`, no overfull boxes in the `.log`.
2. `placed width / native width >= 0.85` at the intended target (section 4).
3. Black, `black!55`, `black!12` only.
4. Every grey element is grey for a stated reason.
5. Any ratio the argument depends on is drawn to scale, with an external label.
6. No em-dash punctuation anywhere in the figure text.
7. Node styles match section 3 so it sits beside the block scheme without clashing.
8. Caption states parameter values and real-vs-schematic.
