# Painting Studio

A desktop companion for painters — color theory, paint mixing, and shading
technique for **watercolor, acrylic, oil, and oil pastel**. Sibling of the
Bargue Block-In app: where that one handles drawing and proportion, this one
handles color.

It makes two documents from the same palette and the same photo: a **palette
board** (a reference sheet of your colours) and a **watercolor painting plan**
(numbered instructions for painting one specific picture). Pick which under
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

## Run

```bash
pip install -r requirements.txt
python main.py
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
   (set a Card width first) to print and keep at the easel.

Sessions auto-restore on launch. Undo/redo covers every edit. Saved presets
capture the board/solver "look" but never overwrite your palette.

## How mixing works

Paint does not mix like light: blue + yellow makes green, not gray. Recipes
are computed in [Mixbox](https://scrtwpns.com/mixbox)'s pigment latent space
(Kubelka–Munk based), searching 1–3 paint combinations from your palette for
the lowest CIEDE2000 distance to the target. Watercolor recipes never use
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
light to dark, and volumes scale only with paper size.

## License note

Pigment mixing uses **Mixbox** © Secret Weapons, licensed
**CC BY-NC 4.0** (non-commercial). The attribution travels on every exported
board. If this app is ever distributed commercially, the mixing engine in
`studio/mixer.py` must be replaced or a commercial Mixbox license obtained
(mixbox@scrtwpns.com).
