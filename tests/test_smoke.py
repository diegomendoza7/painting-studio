#!/usr/bin/env python3
"""Headless smoke test for Painting Studio.

Custom check-based suite (not pytest), mirroring bargue-block-in's. Exercises
the Qt-free core — color math, the Mixbox bridge, the mix solver, medium
rules, value ladders, stage caching — and (in later groups) offscreen Qt
rendering and export.

Run:  QT_QPA_PLATFORM=offscreen python tests/test_smoke.py
"""
from __future__ import annotations

import os
import sys
import tempfile
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from studio import color  # noqa: E402
from studio import mixer  # noqa: E402
from studio.models import Paint, Target  # noqa: E402
from studio.palettes import ACRYLIC_STARTER, OIL_STARTER, OILPASTEL_STARTER, WATERCOLOR_STARTER  # noqa: E402
from studio.params import PRESET_FIELDS, PRESETS, Params, apply_preset, apply_preset_values, preset_to_dict  # noqa: E402
from studio.pipeline import Pipeline  # noqa: E402
from studio.pipeline.loader import LoadedImage, load  # noqa: E402
from studio.pipeline.solver import ACHIEVABLE_DE, solve_targets  # noqa: E402
from studio.scene import ALL_LAYERS, LAYER_RECIPES, LAYER_WHEEL  # noqa: E402
from studio.svg import scene_to_svg  # noqa: E402

PASS, FAIL = "PASS", "FAIL"
_failures = []


def check(cond: bool, msg: str):
    print(f"  [{PASS if cond else FAIL}] {msg}")
    if not cond:
        _failures.append(msg)


# Synthetic reference: four flat patches + seeded noise (a "still life").
PATCHES = [(180, 40, 35), (40, 70, 140), (200, 150, 60), (60, 110, 50)]


def make_still_life(h: int = 360, w: int = 480) -> np.ndarray:
    rng = np.random.default_rng(0)
    img = np.zeros((h, w, 3), np.int16)
    img[: h // 2, : w // 2] = PATCHES[0]
    img[: h // 2, w // 2 :] = PATCHES[1]
    img[h // 2 :, : w // 2] = PATCHES[2]
    img[h // 2 :, w // 2 :] = PATCHES[3]
    img += rng.integers(-4, 5, img.shape, dtype=np.int16)
    return np.clip(img, 0, 255).astype(np.uint8)


def group1_color_math():
    print("\n[1] Color math")
    L, a, b = color.lab((255, 255, 255))
    check(abs(L - 100.0) < 0.1 and abs(a) < 0.5 and abs(b) < 0.5, f"white -> L*≈100 (got {L:.2f})")
    check(abs(color.lstar((119, 119, 119)) - 50.0) < 1.0, "mid grey (119) -> L*≈50")
    check(color.lstar((0, 0, 0)) < 0.5, "black -> L*≈0")
    # Vectorized and scalar paths agree.
    vec = color.rgb_to_lab(np.array([[180, 40, 35]], dtype=np.float64))[0]
    sca = color.lab((180, 40, 35))
    check(max(abs(vec[i] - sca[i]) for i in range(3)) < 1e-9, "numpy and scalar LAB agree")
    rt = color.lab_to_rgb(color.rgb_to_lab(np.array(PATCHES, dtype=np.float64)))
    check(np.max(np.abs(rt.astype(int) - np.array(PATCHES))) <= 1, "LAB round-trip within 1/255")
    # CIEDE2000 against published Sharma test pairs.
    sharma = [
        ((50.0, 2.6772, -79.7751), (50.0, 0.0, -82.7485), 2.0425),
        ((50.0, 3.1571, -77.2803), (50.0, 0.0, -82.7485), 2.8615),
        ((50.0, 2.8361, -74.0200), (50.0, 0.0, -82.7485), 3.4412),
    ]
    for lab1, lab2, expected in sharma:
        got = color.delta_e_2000(lab1, lab2)
        check(abs(got - expected) < 5e-3, f"CIEDE2000 Sharma pair -> {expected} (got {got:.4f})")
    check(color.delta_e_2000((50, 10, -10), (50, 10, -10)) == 0.0, "CIEDE2000 identity is 0")
    d12 = color.delta_e_2000((50.0, 2.6772, -79.7751), (50.0, 0.0, -82.7485))
    d21 = color.delta_e_2000((50.0, 0.0, -82.7485), (50.0, 2.6772, -79.7751))
    check(abs(d12 - d21) < 1e-9, "CIEDE2000 is symmetric")
    # Beer-Lambert dilution.
    base = (140, 60, 40)
    check(color.dilute(base, 1.0) == base, "dilute d=1 is identity")
    check(color.lstar(color.dilute(base, 0.02)) > 95.0, "dilute d->0 approaches paper white")
    ls = [color.lstar(color.dilute(base, d)) for d in (1.0, 0.7, 0.4, 0.15)]
    check(all(ls[i] < ls[i + 1] for i in range(3)), f"dilution L* strictly lightens ({[f'{v:.0f}' for v in ls]})")
    # Temperature.
    check(color.temperature((220, 40, 30)) == "W", "red reads warm")
    check(color.temperature((30, 60, 200)) == "C", "blue reads cool")
    check(color.temperature((128, 128, 128)) == "", "grey reads neutral")


def group2_mixer():
    print("\n[2] Mixer (pymixbox probe)")
    for rgb in [(180, 40, 35), (40, 70, 140), (246, 244, 239)]:
        back = mixer.mix([mixer.latent(rgb)], [1.0])
        check(max(abs(back[i] - rgb[i]) for i in range(3)) <= 3, f"latent round-trip {rgb} -> {back}")
    blue, yellow = (0, 33, 133), (252, 211, 0)
    mixed = mixer.mix([mixer.latent(blue), mixer.latent(yellow)], [0.5, 0.5])
    _L, c, h = color.lch(mixed)
    check(90.0 <= h <= 170.0 and c > 20.0, f"blue + yellow mixes green (hue {h:.0f}°)")
    check(mixer.mix([mixer.latent(blue), mixer.latent(yellow)], [0.5, 0.5]) == mixer.lerp(blue, yellow, 0.5),
          "latent mix at 50/50 equals mixbox.lerp")
    one_sided = mixer.mix([mixer.latent(blue), mixer.latent(yellow)], [1.0, 0.0])
    check(max(abs(one_sided[i] - blue[i]) for i in range(3)) <= 3, "weights (1,0) returns first paint")


def group3_no_photo_pipeline():
    print("\n[3] No-photo pipeline")
    p = Params(manual_targets=(Target(hex="#7a5c3a"),))
    pipe = Pipeline()
    res = pipe.run(p)
    check(res.recomputed == ["extract", "solve", "ladders"],
          f"first run recomputes extract/solve/ladders (got {res.recomputed})")
    check(res.working_rgb is None and res.auto_targets == (), "no photo -> no working image, no auto targets")
    check(len(res.recipes) == 1 and len(res.ladders) == 1, "one recipe + one ladder for one target")
    names = [ly.name for ly in res.scene.layers]
    check(names == list(ALL_LAYERS), f"all named layers present in order (got {names})")
    r = res.recipes[0]
    check(r.target_label == "Color 1", "manual target auto-labeled")
    check(abs(sum(s.weight for s in r.steps) - 1.0) < 1e-6, "recipe weights sum to 1")
    check(all(round(s.weight * 20, 6) == round(s.weight * 20) for s in r.steps),
          "printed weights are 5% multiples")
    check(r.delta_e <= ACHIEVABLE_DE and r.achievable, f"earth target achievable on oil starter (dE {r.delta_e})")
    return pipe, p, res


def group4_determinism(pipe, p, res):
    print("\n[4] Determinism")
    res2 = pipe.run(p)
    check(res2.recomputed == [], "identical params -> nothing recomputed")
    fresh = Pipeline().run(p)
    check(scene_to_svg(fresh.scene) == scene_to_svg(res.scene), "fresh pipeline -> byte-identical SVG")
    check(
        [(s.paint_hex, s.weight) for s in fresh.recipes[0].steps]
        == [(s.paint_hex, s.weight) for s in res.recipes[0].steps],
        "fresh pipeline -> identical recipe weights",
    )


def group5_cache_hierarchy(pipe, p):
    print("\n[5] Cache hierarchy")
    def run(params):
        return pipe.run(params)

    run(p)
    check(run(p.copy(show_wheel=False)).recomputed == [], "board toggle is compose-only")
    check(run(p.copy(show_wheel=False, selected_target=0)).recomputed == [], "selection is compose-only")
    check(run(p.copy(show_wheel=False, selected_target=0, ladder_steps=7)).recomputed == ["ladders"],
          "ladder_steps recomputes ladders only")
    p2 = p.copy(ladder_steps=7, shade_mode="umber")
    check(run(p2).recomputed == ["ladders"], "shade_mode recomputes ladders only")
    p3 = p2.copy(manual_targets=p2.manual_targets + (Target(hex="#aa3344"),))
    check(run(p3).recomputed == ["solve", "ladders"], "adding a target re-solves (never re-extracts)")
    pal = list(p3.palette)
    pal[2] = Paint(name=pal[2].name, hex="#b08030", opacity=pal[2].opacity)
    check(run(p3.copy(palette=tuple(pal))).recomputed == ["solve", "ladders"],
          "editing a paint's color re-solves")
    pal2 = list(p3.copy(palette=tuple(pal)).palette)
    pal2[2] = Paint(name="Renamed Ochre", hex=pal2[2].hex, opacity=pal2[2].opacity)
    check(run(p3.copy(palette=tuple(pal2))).recomputed == [],
          "renaming a paint recomputes NOTHING (names resolve at compose)")
    check(run(p3.copy(palette=tuple(pal2), max_paints=2)).recomputed == ["solve", "ladders"],
          "max_paints re-solves")
    check(run(p3.copy(palette=tuple(pal2), max_paints=2, medium="acrylic")).recomputed == ["solve", "ladders"],
          "switching medium re-solves")


def group6_photo_extraction():
    print("\n[6] Photo extraction")
    img = make_still_life()
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "still_life.png")
        from PIL import Image

        Image.fromarray(img).save(path)
        loaded = load(path)
        check(loaded.rgb.shape == img.shape, "loader returns the PNG as RGB")

    pipe = Pipeline()
    pipe.set_source(LoadedImage(rgb=img, source_path="synthetic"))
    p = Params(extract_n=4)
    res = pipe.run(p)
    check(res.recomputed == ["normalize", "extract", "solve", "ladders"],
          f"first photo run recomputes all stages (got {res.recomputed})")
    check(len(res.auto_targets) == 4 and len(res.recipes) == 4, "extract_n=4 -> 4 auto targets + recipes")
    for patch in PATCHES:
        best = min(
            color.delta_e_2000(color.lab(patch), color.lab(color.hex_to_rgb(t.hex)))
            for t in res.auto_targets
        )
        check(best <= 12.0, f"patch {patch} matched by an auto target (dE {best:.1f})")
    pipe_b = Pipeline()
    pipe_b.set_source(LoadedImage(rgb=img, source_path="synthetic"))
    res_b = pipe_b.run(p)
    check([t.hex for t in res_b.auto_targets] == [t.hex for t in res.auto_targets],
          "extraction is deterministic across pipelines")
    check(pipe.run(p.copy(extract_n=6)).recomputed == ["extract", "solve", "ladders"],
          "extract_n re-extracts but never re-normalizes")
    check(pipe.run(p.copy(extract_n=6, manual_targets=(Target(hex="#888888", source="eyedrop"),))).recomputed
          == ["solve", "ladders"], "an eyedropper pick only re-solves")
    check(pipe.run(p.copy(extract_n=6, manual_targets=(Target(hex="#888888", source="eyedrop"),),
                          max_edge=640)).recomputed
          == ["normalize", "extract", "solve", "ladders"], "max_edge invalidates everything")
    check(res.layout.photo_rect is not None and res.layout.photo_img_size == (480, 360),
          "photo placed on the board with source size recorded")


def group7_solver_accuracy():
    print("\n[7] Solver accuracy + budget")
    ultra = next(p for p in OIL_STARTER if p.name == "Ultramarine Blue")
    umber = next(p for p in OIL_STARTER if p.name == "Burnt Umber")
    made = mixer.mix([mixer.latent_hex(ultra.hex), mixer.latent_hex(umber.hex)], [0.6, 0.4])
    recipes = solve_targets([Target(hex=color.rgb_to_hex(made), label="made")], OIL_STARTER, "oil", 3)
    r = recipes[0]
    check(r.delta_e <= 2.0, f"mixbox-constructed 60/40 target recovered (dE {r.delta_e})")
    hexes = {s.paint_hex for s in r.steps}
    check(hexes <= {ultra.hex, umber.hex}, f"recovered recipe uses the constructing paints ({hexes})")
    # A palette paint as the target -> the single-paint recipe wins.
    recipes = solve_targets([Target(hex=ultra.hex)], OIL_STARTER, "oil", 3)
    check(len(recipes[0].steps) == 1 and recipes[0].steps[0].paint_hex == ultra.hex
          and recipes[0].delta_e <= 0.5, "palette paint target -> single-paint recipe")
    # Neon green is outside an earth-and-primaries gamut.
    recipes = solve_targets([Target(hex="#39ff14")], OIL_STARTER, "oil", 3)
    check(not recipes[0].achievable and recipes[0].delta_e > ACHIEVABLE_DE,
          f"neon target flagged out of gamut (dE {recipes[0].delta_e})")
    check(len(recipes[0].steps) >= 1, "nearest mix still reported for out-of-gamut targets")
    # Performance budget: 8 targets on the 12-paint watercolor palette.
    targets = [Target(hex=h) for h in
               ("#c46a4a", "#7a8b5a", "#4a5d8a", "#d9b98a", "#8a4a6a", "#5a7a7a", "#b98a3a", "#6a4a3a")]
    t0 = time.perf_counter()
    recipes = solve_targets(targets, WATERCOLOR_STARTER, "watercolor", 3)
    dt = time.perf_counter() - t0
    check(len(recipes) == 8, "all 8 targets solved")
    check(dt <= 2.0, f"8 targets x 12 paints within budget ({dt:.2f}s)")


def group8_medium_constraints():
    print("\n[8] Medium constraints")
    # Watercolor: white never appears; light targets dilute instead.
    wc_pal = WATERCOLOR_STARTER + (Paint("Chinese White", "#f4f2ec", is_white=True),)
    targets = [Target(hex="#e8c9a0"), Target(hex="#c46a4a"), Target(hex="#dce6ee")]
    recipes = solve_targets(targets, wc_pal, "watercolor", 3)
    check(all("#f4f2ec" not in {s.paint_hex for s in r.steps} for r in recipes),
          "watercolor recipes never include white paint")
    light = recipes[0]
    check(light.dilution < 0.95, f"light watercolor target lightened by dilution (d={light.dilution})")
    check("wash" in light.note or "water" in light.note, "watercolor note speaks in washes")
    # Oil pastel: ordered layering, pressure words, no weights-sum semantics.
    base = next(p for p in OILPASTEL_STARTER if p.name == "Brown")
    top = next(p for p in OILPASTEL_STARTER if p.name == "Sky Blue")
    made = mixer.lerp(color.hex_to_rgb(base.hex), color.hex_to_rgb(top.hex), 0.65)
    recipes = solve_targets([Target(hex=color.rgb_to_hex(made))], OILPASTEL_STARTER, "oilpastel", 2)
    r = recipes[0]
    check(r.layering and len(r.steps) == 2, "oil pastel solves as a two-stick layering")
    check(r.steps[0].paint_hex == base.hex and r.steps[1].paint_hex == top.hex,
          "layering keeps base -> top order")
    check("pressure" in r.note, f"pastel note gives pressure guidance ({r.note!r})")
    # Oil: white-heavy tints carry the chalkiness warning.
    recipes = solve_targets([Target(hex="#f0d8d2")], OIL_STARTER, "oil", 3)
    white_w = sum(s.weight for s in recipes[0].steps if s.paint_hex == "#f6f4ef")
    check(white_w > 0.4, f"pastel pink needs a white-heavy mix ({white_w:.2f})")
    check("chalk" in recipes[0].note, "oil note warns about chalkiness")
    # Acrylic: dries-darker reminder.
    recipes = solve_targets([Target(hex="#7a5c3a")], ACRYLIC_STARTER, "acrylic", 3)
    check("darker" in recipes[0].note, "acrylic note reminds it dries darker")


def group9_ladders():
    print("\n[9] Value ladders")
    pipe = Pipeline()
    p = Params(manual_targets=(Target(hex="#9c5030"),), ladder_steps=7)
    res = pipe.run(p)
    lad = res.ladders[0]
    check(len(lad.steps) == 7, "ladder has ladder_steps steps")
    ls = [color.lstar(color.hex_to_rgb(s.hex)) for s in lad.steps]
    check(all(ls[i] > ls[i + 1] for i in range(len(ls) - 1)),
          f"ladder strictly light -> dark ({[f'{v:.0f}' for v in ls]})")
    check(any("recipe mix" in s.recipe_text for s in lad.steps), "ladder includes the mix itself")
    check(lad.note != "", "ladder carries a medium technique note")
    # Watercolor: lighter steps stay in the family (dilution, not white shift).
    pw = Params(medium="watercolor", palette=WATERCOLOR_STARTER,
                manual_targets=(Target(hex="#9c5030"),), ladder_steps=5)
    resw = Pipeline().run(pw)
    ladw = resw.ladders[0]
    base_i = next(i for i, s in enumerate(ladw.steps) if s.hex == resw.recipes[0].mixed_hex)
    bh = color.lch(color.hex_to_rgb(ladw.steps[base_i].hex))[2]
    for s in ladw.steps[:base_i]:
        hh = color.lch(color.hex_to_rgb(s.hex))[2]
        dh = min(abs(hh - bh), 360 - abs(hh - bh))
        check(dh <= 30.0, f"watercolor light step keeps the wash hue (Δh {dh:.0f}°)")
    check(all("thin" in s.recipe_text for s in ladw.steps[:base_i]),
          "watercolor lighter steps are dilutions")
    # shade_mode=black picks the lowest-chroma dark (acrylic has Mars Black).
    pb = Params(medium="acrylic", palette=ACRYLIC_STARTER,
                manual_targets=(Target(hex="#c46a4a"),), shade_mode="black")
    resb = Pipeline().run(pb)
    dark_paints = {s.paint_hex for s in resb.ladders[0].steps if s.paint_hex and "+" in s.recipe_text
                   and color.lstar(color.hex_to_rgb(s.hex)) < color.lstar(color.hex_to_rgb(resb.recipes[0].mixed_hex))}
    check("#2b2b2b" in dark_paints, f"shade_mode=black darkens with Mars Black ({dark_paints})")


def group10_harmony_wheel():
    print("\n[10] Harmony / wheel / gamut")
    base = Params(manual_targets=(Target(hex="#c46a4a"),))  # warm anchor via first target
    counts = {}
    for scheme in ("none", "complementary", "analogous", "triadic", "split"):
        res = Pipeline().run(base.copy(harmony_scheme=scheme))
        wheel = res.scene.layer(LAYER_WHEEL)
        counts[scheme] = len(wheel.strokes)
        check(res.layout.wheel is not None, f"wheel geometry exported for hit-testing ({scheme})")
    check(counts["complementary"] > counts["none"], "complementary adds overlay strokes")
    # complementary = anchor ring + 1 line + 1 partner marker; analogous/split = 2 each.
    check(counts["analogous"] == counts["complementary"] + 2, "analogous has a second partner")
    check(counts["split"] == counts["analogous"], "split mirrors analogous stroke count")
    check(counts["triadic"] == counts["analogous"] + 1, "triadic closes the triangle")
    res_g = Pipeline().run(base.copy(show_gamut=True))
    res_ng = Pipeline().run(base.copy(show_gamut=False))
    wl_g = res_g.scene.layer(LAYER_WHEEL)
    wl_ng = res_ng.scene.layer(LAYER_WHEEL)
    check(len(wl_g.strokes) == len(wl_ng.strokes) + 1, "gamut hull polygon present when enabled")


def group11_board_layout():
    print("\n[11] Board compose + hit-test layout")
    img = make_still_life()
    pipe = Pipeline()
    pipe.set_source(LoadedImage(rgb=img, source_path="synthetic"))
    p = Params(extract_n=4, manual_targets=(Target(hex="#7a5c3a"),))
    res = pipe.run(p)
    check(len(res.layout.swatch_rects) == len(p.palette), "one hit-rect per palette swatch")
    check(len(res.layout.target_rects) == len(res.recipes), "one hit-rect per target row")
    ys = [r[0][1] for r in res.layout.target_rects]
    check(all(ys[i] + res.layout.target_rects[i][0][3] <= ys[i + 1] + 1 for i in range(len(ys) - 1)),
          "target rows vertically ordered, non-overlapping")
    rx, ry, rw, rh = res.layout.photo_rect
    check(rx >= 0 and ry >= 0 and rx + rw <= res.scene.width and ry + rh <= res.scene.height,
          "photo rect inside the scene")
    for field, layer in (("show_palette", "palette-swatches"), ("show_wheel", LAYER_WHEEL),
                         ("show_recipes", LAYER_RECIPES), ("show_notes", "notes")):
        res_off = pipe.run(p.copy(**{field: False}))
        check(not res_off.scene.layer(layer).visible, f"{field}=False hides its layer")
        check(res_off.scene.height < res.scene.height, f"{field}=False compacts the board")
    res_sel = pipe.run(p.copy(selected_target=0))
    n_recipes_strokes = len(res.scene.layer(LAYER_RECIPES).strokes)
    check(len(res_sel.scene.layer(LAYER_RECIPES).strokes) == n_recipes_strokes + 1,
          "selected target draws a highlight outline")


def group12_svg_export():
    print("\n[12] SVG export")
    img = make_still_life()
    pipe = Pipeline()
    pipe.set_source(LoadedImage(rgb=img, source_path="synthetic"))
    p = Params(extract_n=4, board_title="Still life & fruit <study>")
    svg = scene_to_svg(pipe.run(p).scene)
    import xml.dom.minidom

    try:
        xml.dom.minidom.parseString(svg)
        well_formed = True
    except Exception:
        well_formed = False
    check(well_formed, "SVG is well-formed XML (board title with & < > escaped)")
    for name in ALL_LAYERS:
        check(f'inkscape:label="{name}"' in svg, f"Inkscape layer '{name}' present")
    check("<image" in svg, "photo embedded as <image> when shown")
    svg_no_photo = scene_to_svg(pipe.run(p.copy(show_photo=False)).scene)
    check("<image" not in svg_no_photo, "no <image> when the photo region is hidden")
    check("Mixbox" in svg and "CC BY-NC" in svg, "Mixbox attribution travels with the export")


def group13_offscreen_render():
    print("\n[13] Offscreen Qt render")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    app.setOrganizationName("StudioSmokeTest")
    app.setApplicationName("StudioSmokeTest")
    from studio.ui.render import render_scene

    pipe = Pipeline()
    pipe.set_source(LoadedImage(rgb=make_still_life(), source_path="synthetic"))
    p = Params(extract_n=4)
    res = pipe.run(p)
    image = render_scene(res.scene, 1.0)
    check(image.width() == res.scene.width and image.height() == res.scene.height,
          "raster matches scene size")
    bg = image.pixelColor(2, 2)
    check(bg.name() == p.background, f"background pixel is the card color ({bg.name()})")
    # The center of each swatch's masstone rect is exactly the paint color.
    ok = True
    for (rx, ry, rw, rh), idx in res.layout.swatch_rects:
        c = image.pixelColor(int(rx + rw / 2), int(ry + rh / 2))
        want = p.palette[idx].hex
        got = c.name()
        if max(abs(int(want[i:i + 2], 16) - int(got[i:i + 2], 16)) for i in (1, 3, 5)) > 2:
            ok = False
            break
    check(ok, "swatch-center pixels match the paint hexes (raster agrees with layout)")
    half = render_scene(res.scene, 0.5)
    check(abs(half.width() - res.scene.width / 2) <= 1, "render scale honored")


def group14_to_scale_export():
    print("\n[14] To-scale export")
    from studio.pdf_export import MAX_RASTER_EDGE, raster_scale, write_scale_pdf
    from studio.ui.render import render_scene

    p = Params(manual_targets=(Target(hex="#7a5c3a"),), card_w=11.0)
    res = Pipeline().run(p)
    check(res.scene.inches_per_px > 0, "card width sets the physical scale")
    svg = scene_to_svg(res.scene)
    check('width="11in"' in svg, "SVG carries the physical width (11in)")
    p_cm = p.copy(card_w=28.0, canvas_units="cm")
    svg_cm = scene_to_svg(Pipeline().run(p_cm).scene)
    check('width="28cm"' in svg_cm, "cm units label the SVG in centimetres")
    ipp = res.scene.inches_per_px
    scale = raster_scale(res.scene.width, res.scene.height, 0, ipp)
    longest = max(res.scene.width, res.scene.height) * scale
    check(longest <= MAX_RASTER_EDGE + 1, "raster cap respected")
    with tempfile.TemporaryDirectory() as td:
        png_path = os.path.join(td, "board.pdf")
        image = render_scene(res.scene, min(scale, 0.5))
        from PySide6.QtCore import QBuffer, QByteArray

        ba = QByteArray()
        buf = QBuffer(ba)
        buf.open(QBuffer.WriteOnly)
        image.save(buf, "PNG")
        buf.close()
        write_scale_pdf(ba.data(), 11.0, 11.0 * res.scene.height / res.scene.width, png_path)
        with open(png_path, "rb") as fh:
            head = fh.read(5)
        check(head == b"%PDF-", "PDF export starts with %PDF")


def group15_session_roundtrip():
    print("\n[15] Session round-trip")
    from studio.session import decode_session, encode_session

    pal = OIL_STARTER[:3] + (Paint("My Mauve", "#a06a8a", opacity="semi", staining=True),)
    p = Params(
        medium="acrylic", palette=pal,
        manual_targets=(Target(hex="#aa3344", label="apple"), Target(hex="#445566", source="eyedrop")),
        extract_n=9, ladder_steps=6, board_title="Test", card_w=12.5, canvas_units="cm",
    )
    blob = encode_session(path="/tmp/x.png", page=2, dpi=300, params=p)
    state = decode_session(blob)
    check(state is not None and state["params"] == p,
          "decode(encode(params)) == params (Paint/Target tuples rebuilt)")
    check(state["params"].solve_key() == p.solve_key(), "solve cache key survives a restart")
    check(state["path"] == "/tmp/x.png" and state["page"] == 2 and state["dpi"] == 300,
          "photo path/page/dpi round-trip")
    check(decode_session("") is None, "empty blob -> fresh start")
    check(decode_session("not json{") is None, "corrupt blob -> fresh start")
    check(decode_session('{"schema": 99, "params": {}}') is None, "wrong schema -> fresh start")


def group16_presets():
    print("\n[16] Presets")
    p = Params()
    for name, changes in PRESETS.items():
        q = apply_preset(p, name)
        check(q.medium == changes["medium"] and q.palette == changes["palette"],
              f"built-in '{name}' switches medium + palette")
        check(len(Pipeline().run(q.copy(manual_targets=(Target(hex='#8a6a4a'),))).recipes) == 1,
              f"built-in '{name}' runs end-to-end")
    # User presets: only PRESET_FIELDS travel.
    src = Params(max_paints=2, ladder_steps=7, shade_mode="umber", harmony_scheme="triadic",
                 show_wheel=False, extract_n=10, medium="watercolor",
                 palette=WATERCOLOR_STARTER, manual_targets=(Target(hex="#aa3344"),),
                 board_title="KEEP")
    saved = preset_to_dict(src)
    check(set(saved) == set(PRESET_FIELDS), "saved preset captures exactly PRESET_FIELDS")
    dst = apply_preset_values(Params(), saved)
    check(dst.max_paints == 2 and dst.ladder_steps == 7 and not dst.show_wheel,
          "saved look applies")
    check(dst.medium == "oil" and dst.palette == Params().palette and dst.manual_targets == ()
          and dst.board_title == "", "user preset never touches medium/palette/targets/title")


def group17_ui_smoke():
    print("\n[17] UI smoke")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    app.setOrganizationName("StudioSmokeTest")
    app.setApplicationName("StudioSmokeTest")
    from PySide6.QtCore import QSettings

    QSettings().clear()  # keep this run hermetic
    from studio.ui.controls import ControlsPanel, FloatSlider
    from studio.ui.main_window import MainWindow

    sl = FloatSlider(0.002, 0.06, 0.012, decimals=4, log=True)
    sl.setValue(0.01)
    check(abs(sl.value() - 0.01) / 0.01 < 0.02, "FloatSlider log mapping round-trips")
    sl.setValue(0.002)
    lo = sl.value()
    sl.setValue(0.06)
    hi = sl.value()
    check(abs(lo - 0.002) < 1e-9 and abs(hi - 0.06) < 1e-9, "FloatSlider hits its bounds")

    p = Params(medium="watercolor", palette=WATERCOLOR_STARTER, extract_n=9, max_paints=2,
               ladder_steps=6, shade_mode="umber", harmony_scheme="triadic",
               anchor_hex="#aa3344", show_gamut=False, show_photo=False,
               board_title="My board", card_w=12.5, canvas_units="cm")
    panel = ControlsPanel(p)
    panel.apply(p)
    check(panel.collect(p) == p, "controls collect(apply(p)) == p")

    win = MainWindow()
    deadline = time.monotonic() + 10.0
    while win._result is None and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(0.01)
    check(win._result is not None, "MainWindow renders the default board on launch")
    check(win.act_svg.isEnabled(), "exports enabled after the first render")

    # Eyedropper: photo-rect click maps to the right source pixel.
    img = make_still_life()
    win._worker.pipeline.set_source(LoadedImage(rgb=img, source_path="synthetic"))
    win._render_now()
    deadline = time.monotonic() + 10.0
    while (win._result is None or win._result.working_rgb is None) and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(0.01)
    check(win._result is not None and win._result.layout.photo_rect is not None,
          "photo board rendered")
    win.act_eyedrop.setChecked(True)
    rx, ry, rw, rh = win._result.layout.photo_rect
    iw, ih = win._result.layout.photo_img_size
    sx = rx + (120.0 / iw) * rw   # center of the top-left (red) patch
    sy = ry + (90.0 / ih) * rh
    n_before = len(win.params.manual_targets)
    win._on_preview_click(sx, sy, 1)
    check(len(win.params.manual_targets) == n_before + 1, "eyedropper click adds a target")
    picked = color.hex_to_rgb(win.params.manual_targets[-1].hex)
    de = color.delta_e_2000(color.lab(PATCHES[0]), color.lab(picked))
    check(de <= 4.0, f"eyedropped color matches the patch under the cursor (dE {de:.1f})")
    win._do_undo()
    check(len(win.params.manual_targets) == n_before, "undo removes the eyedropped target")

    win._thread.quit()
    win._thread.wait(3000)
    QSettings().clear()


def main() -> int:
    group1_color_math()
    group2_mixer()
    pipe, p, res = group3_no_photo_pipeline()
    group4_determinism(pipe, p, res)
    group5_cache_hierarchy(pipe, p)
    group6_photo_extraction()
    group7_solver_accuracy()
    group8_medium_constraints()
    group9_ladders()
    group10_harmony_wheel()
    group11_board_layout()
    group12_svg_export()
    group13_offscreen_render()
    group14_to_scale_export()
    group15_session_roundtrip()
    group16_presets()
    group17_ui_smoke()

    print(f"\n{'=' * 60}")
    if _failures:
        print(f"{len(_failures)} FAILURE(S):")
        for f in _failures:
            print(f"  - {f}")
        return 1
    print("All checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
