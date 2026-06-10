# Painting Studio

A desktop companion for painters — color theory, paint mixing, and shading
technique for **watercolor, acrylic, oil, and oil pastel**. Sibling of the
Bargue Block-In app: where that one handles drawing and proportion, this one
handles color.

The document it produces is a **palette board**: a printable study sheet of

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
round-trips, and an offscreen run of the full UI.

## License note

Pigment mixing uses **Mixbox** © Secret Weapons, licensed
**CC BY-NC 4.0** (non-commercial). The attribution travels on every exported
board. If this app is ever distributed commercially, the mixing engine in
`studio/mixer.py` must be replaced or a commercial Mixbox license obtained
(mixbox@scrtwpns.com).
