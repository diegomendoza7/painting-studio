# Painting Studio

A desktop companion for painters — color theory, paint mixing, and shading
technique for **watercolor, acrylic, oil, and oil pastel**. Sibling of the
Bargue Block-In app: where that one handles drawing and proportion, this one
handles color.

It makes three documents from the same palette and the same photo: a **palette
board** (a reference sheet of your colours), a **watercolor painting plan**
(numbered instructions for painting one specific picture), and a **mosaic
guide** (the photo as tesserae, painted one square at a time). Pick which under
*Document* in the right-hand panel.

## Palette board

A printable study sheet of

- **Your palette** — the paints you actually own, as swatch cards with
  medium-appropriate tint strips (dilution washes for watercolor, white tints
  for oil/acrylic, pressure tints for oil pastel), value numbers, warm/cool
  letters, and trait glyphs (Transparent/Semi/Opaque, Staining, Granulating).
- **Color wheel & harmony** — your paints plotted by hue and chroma, a
  harmony overlay (complementary / analogous / triadic / split) around an
  anchor color, and the *mixable gamut* your palette can actually reach.
- **Value scale** — your paints sorted light to dark.
- **Targets & mixing recipes** — colors extracted from a reference photo
  (and/or hand-picked), each with the best 2–3 paint mix from *your*
  palette: proportions, the honest ΔE of the printed recipe, and an
  out-of-gamut flag with the nearest achievable mix when a color can't be hit.
- **Value ladders** — light→dark shading steps for every target with
  per-step recipes in the medium's own language: dilution for watercolor,
  tints/transparent darks for oil and acrylic, stick layering orders with
  pressure notes for oil pastel.
- **Technique notes** — per-medium guidance (fat over lean, glazing,
  wet-on-wet, burnishing…).

## Watercolor painting plan

Built for someone learning watercolor. Load a photo and the plan works out how
to paint *that picture* with the paints you actually own:

- **A numbered map** — the photo beside the same shapes as they are planned to
  be painted. Every shape is a *passage*, numbered in painting order.
- **A value plan** — the picture's values quantized into tiers, which is
  literally how many layers of wash deep the painting goes.
- **Whites to reserve** — the shapes light enough to leave as bare paper,
  named before the first brushstroke, since watercolor has no white paint.
- **The washes, in order** — one block per pass: the mix as a ratio of parts
  from your palette, how much wash to mix in millilitres and brush-loads, the
  paint-to-water strength, wet-on-wet or wet-on-dry, which edges go crisp and
  which shapes to stop against, why the pass exists, and a drying break.
  Passes run strictly light to dark, because a transparent medium only ever
  goes darker.
- **A test strip** — every mix as a swatch to paint on scrap and judge *dry*,
  since watercolor lightens 20–30% as it dries.
- **Tips and a glossary** — one lesson attached to each pass, never repeated,
  plus the vocabulary the plan uses.

Set your **paper size** in the plan controls: that is what turns "27% of the
picture" into "mix about 11 ml". Volumes are deliberately generous — a wash
re-mixed halfway through never matches. Click any wash to highlight its shapes
on the map.

## Mosaic guide

Roman/Byzantine mosaic as a beginner's watercolor exercise. The photo is cut
into tesserae and reduced to a small stock of colours, and each tessera is one
flat square of paint — no edges to control, no wet-in-wet timing, and the grout
is just bare paper you leave between the squares.

- **Andamento — courses that follow the drawing.** The tesserae are not on a
  lattice. The picture's broad value masses are found first, a distance field is
  grown *inside* each of them, and the stones are marched along its level sets,
  so a course hugs each contour and the next stacks behind it — *opus
  vermiculatum*, the way a mosaicist works. The first course is held back from
  its contour so an unbroken interstice runs along every edge. That is what
  makes the mosaic read as a picture **with the colour taken away**; on a plain
  grid it never can.
- **Stones cut to fit.** Courses are laid at full size first, then filling
  passes cut progressively smaller tesserae into whatever gaps are left, the way
  a mosaicist shapes an odd stone rather than leaving a hole. Without them any
  gap narrower than one stone stays bare and reads as a deliberate white shape;
  with them the mosaic covers the whole picture and the only bare paper is a
  thin, even grout line.
- **The mosaic in your paints** — the photo beside the mosaic rendered in the
  colours your palette can actually reach, with hand-laid tesserae and grout.
- **A cartoon** — every stone outlined at true size with its colour number
  inside. A flowing course cannot be addressed by row and column, so you trace
  the stones, which is how a mosaic is really transferred. (Switch *Layout* to
  **Square grid** if you would rather rule a lettered lattice onto paper.)
- **A legend** — per colour: the photo's colour beside what your palette
  reaches, the mix as a ratio of parts, the wash strength, tile count and
  coverage, and the honest ΔE. Colours are numbered **lightest first**, so
  painting them in order is also painting light to dark.
- **Export Cartoon (to scale)** — prints at your chosen tessera size (12 mm by
  default), tiled across as many sheets as it needs, with crop marks and
  "row 1 / col 2" labels so they assemble unambiguously. Tape them together and
  paint straight from them, or trace them onto your paper.

*Irregularity* is measured against the *grout*, not the stone — two neighbours
can each shift a corner by at most half the gap, so a tessera can never leave
its own footprint. A wider grout is therefore what buys a more hand-cut look.
The grout itself is floored at a visible minimum, because two wet watercolor
squares that touch bleed into one.

## Easel — colour matching while you paint

The window you keep open beside the canvas. Open it with the **Easel** button,
or go straight there:

```bash
python main.py --easel
```

- **Click the reference, get the colour big.** The colour fills the window on
  a neutral mid-grey surround, the one background that doesn't push a colour
  warmer, cooler, lighter or darker. Click again to go back. Three ways to pick:
  - **Area — average**: selects the patch of similar colour you clicked
    (outlined on the photo) and averages it — the colour you'd mix for that
    shape. **Size** makes it smaller or larger than the automatic choice.
  - **Eyedropper — exact pixel**: the colour of the one pixel under the cursor,
    no averaging, read from the full-resolution photo. A **loupe** magnifies
    the pixels round the cursor so the pick lands where you mean, and the point
    is ringed on the photo afterwards.
  - **Small spot — average**: a small circle, for details an area would swallow.

  **Option-click** uses the eyedropper while picking areas (and the area while
  using the eyedropper); **E** switches between the two.
- **What to mix, from your paints.** The recipe as parts, the closest colour
  your palette actually reaches (shown under the target, with how close it is
  in words), and numbered steps **in the order that works**: white first for
  tints (colour goes into white, never the reverse), the strongest pigments —
  Phthalo, Alizarin — last and a little at a time, because you can't take them
  back out. Oil and acrylic mixes may use four paints: muted neutrals, which
  are most of what you mix, usually need white plus a warm, a cool and a
  modifier.
- **How much.** From your canvas size and the region's share of the picture:
  millilitres, a pile you can picture (a pea, a hazelnut, a walnut), and
  centimetres squeezed from a standard tube — per paint.
- **V** shows value only, on the swatch and the photo. Value is the part of a
  colour that's easiest to get wrong and matters most.
- **History, by painting.** Every swatch is saved as you go, with a crop of
  where it came from, and can be named, tagged and annotated. Search one
  painting or all of them by name, notes, tag, paint or painting. **One swatch
  per area:** clicking an area that's already in the history opens that swatch
  rather than adding a copy (a different Size setting, or Option-click, still
  makes a new one).
- **Named palettes.** Keep several (a full kit, a Zorn limited palette…); each
  painting remembers which it mixes from. Changing a palette never rewrites an
  old swatch — **Re-mix** does that when you ask.

Everything lives in `~/Documents/Painting Studio/` as plain JSON and PNG: one
folder per painting with a *copy* of its reference photo, so moving or deleting
the original never breaks it. iPhone HEIC photos are converted on import with
macOS's built-in `sips`.

Keys: **Space** back to the photo · **Esc** back, then clear the selection ·
**⌘D** clear the selection (so is clicking the margin around the photo) ·
**V** value · **E** Area ↔ Eyedropper · **⌫** delete swatch · **⌘N** new painting ·
**⌘F** search · **⌃⌘F** full screen.

## Projects

One project is one painting: its reference photo, the document's settings
(board, plan or mosaic, palette, every control) and the Easel's swatches, kept
together so you can leave a painting half-done, start another, and come back to
either exactly as you left it.

- **File → New Project** (⌘N), **Open…** (⌘O, with a thumbnail of each
  painting), **Open Recent** (also the ▾ beside the toolbar's Open), **Save**
  (⌘S), **Save As…** (⇧⌘S — a full copy, swatches included), and
  **Reference Photo…** (⇧⌘O) to change the photo.
- The title shows the project's name, and a dot on the close button when it has
  unsaved changes. Switching project or quitting asks first — Save, Don't Save,
  or Cancel. Clicking to highlight a colour or wash is not an edit.
- Launching reopens the last project. Work you never saved stays in the untitled
  workspace, which restores itself as before.
- The Easel and the document window share projects: the Easel button opens the
  Easel on the painting you have open, a painting started in the Easel opens here
  with its palette, and each window saves only its own half, so neither can undo
  the other's work — even when they run as two separate apps.

Projects live in `~/Documents/Painting Studio/projects/`, the same folders as
the Easel's paintings. The photo is copied in on the first save.

## Run

```bash
pip install -r requirements.txt
python main.py            # documents: board, painting plan, mosaic
python main.py --easel    # straight to the easel
```

## Workflow

1. Pick a **medium** (or a starter preset — it loads a sensible palette).
2. Edit the **palette** in the left panel to match your actual tubes/sticks.
3. Add **target colors**: open a reference photo (the dominant palette is
   extracted automatically), use the **Eyedropper** to probe spots on the
   photo, or **Add Color…** by hand.
4. Read the recipes and ladders; click a target row to anchor the harmony
   wheel on it; click a paint swatch to anchor on that paint instead.
5. **Export** — SVG (layered, Inkscape-friendly), PNG, or a to-scale PDF
   (set a Card width first) to print and keep at the easel. PDFs are true
   vector, so text and edges stay sharp at any size. In mosaic mode,
   **Export Cartoon** prints the numbered stones at true tessera size across
   however many sheets that takes.

The last project reopens on launch. Undo/redo covers every edit. Saved presets
capture the board/solver "look" but never overwrite your palette.

## How mixing works

Paint does not mix like light: blue + yellow makes green, not gray. Recipes
are computed in [Mixbox](https://scrtwpns.com/mixbox)'s pigment latent space
(Kubelka–Munk based), searching 1–4 paint combinations from your palette for
the lowest CIEDE2000 distance to the target. Four-paint mixes are tried on top
of the best three-paint ones, which keeps a click fast on a large palette. Watercolor recipes never use
white — lightness comes from dilution over the paper, modeled with
Beer–Lambert. Oil pastel "recipes" are surface layering orders, since the
medium mixes on the support, not the palette.

All mixes are estimates — test on scrap before committing.

## Tests

```bash
QT_QPA_PLATFORM=offscreen python tests/test_smoke.py
```

A check-based headless suite: color math against published CIEDE2000 test
pairs, solver accuracy and determinism, per-medium constraint enforcement,
stage-cache behavior, board layout/hit-testing, SVG/PNG/PDF export, session
round-trips, and an offscreen run of the full UI. For the painting plan it
also locks the invariants that make the plan safe to follow — passages cover
the image exactly, every one is painted once or reserved, washes run strictly
light to dark, and volumes scale only with paper size. For the mosaic it locks
that every tessera stays inside its own footprint at any grout and irregularity
setting, that colours are numbered lightest first, that the to-scale sheet
prints one cell per tessera, and that the tile plan pages a wall-sized grid
correctly. The andamento is tested on a synthetic disc, where a course that
follows the drawing *must* run tangential to the circle — horizontal bands
could not tell andamento apart from a lattice. Coverage is measured by
rasterizing the drawn stones, so a regression that reopens bare patches fails
rather than merely looking wrong.

## License note

Pigment mixing uses **Mixbox** © Secret Weapons, licensed
**CC BY-NC 4.0** (non-commercial). The attribution travels on every exported
board. If this app is ever distributed commercially, the mixing engine in
`studio/mixer.py` must be replaced or a commercial Mixbox license obtained
(mixbox@scrtwpns.com).

## About this history

Built locally between June and September 2026. Uploaded to GitHub on October 6, 2026. Commit dates reflect when the work was done on my machine. History is reconstructed from file timestamps and session logs, so early intermediate versions may be missing.

The first commit is the exact snapshot staged in git on June 10.
