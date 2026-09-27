#!/usr/bin/env python3
"""Headless smoke test for Painting Studio.

Custom check-based suite (not pytest), mirroring bargue-block-in's. Exercises
the Qt-free core — color math, the Mixbox bridge, the mix solver, medium
rules, value ladders, stage caching — and (in later groups) offscreen Qt
rendering and export.

Run:  QT_QPA_PLATFORM=offscreen python tests/test_smoke.py
"""
from __future__ import annotations

import math
import os
import sys
import tempfile
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Keep every library the suite creates out of ~/Documents/Painting Studio.
os.environ["PAINTING_STUDIO_LIBRARY"] = tempfile.mkdtemp(prefix="studio-smoke-lib-")

from studio import color  # noqa: E402
from studio import mixer  # noqa: E402
from studio.models import Paint, Target  # noqa: E402
from studio.palettes import ACRYLIC_STARTER, OIL_STARTER, OILPASTEL_STARTER, WATERCOLOR_STARTER  # noqa: E402
from studio.params import DOC_BOARD, DOC_MOSAIC, DOC_PLAN, PRESET_FIELDS, PRESETS, Params, apply_preset, apply_preset_values, preset_to_dict  # noqa: E402
from studio.pipeline import Pipeline  # noqa: E402
from studio.pipeline.loader import LoadedImage, load  # noqa: E402
from studio.pipeline.mosaic_doc import build_grid_sheet  # noqa: E402
from studio.pipeline.solver import ACHIEVABLE_DE, solve_targets  # noqa: E402
from studio.pdf_export import PAPER_SIZES_IN, plan_tiles  # noqa: E402
from studio.pipeline import mosaic_doc, mosaic_flow, mosaic_grid, plan_regions, plan_schedule  # noqa: E402
from studio.scene import ALL_LAYERS, LAYER_MOSAIC_LEGEND, LAYER_PLAN_MAP, LAYER_PLAN_PASSES, LAYER_RECIPES, LAYER_WHEEL, MOSAIC_LAYERS, PLAN_LAYERS  # noqa: E402
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


# A high-key "landscape": a near-white sky, two mid masses and a dark, in
# nested horizontal bands, so the plan's tiers and ordering are predictable.
PLAN_BANDS = [(238, 236, 230), (152, 176, 206), (188, 150, 92), (52, 58, 44)]


def make_plan_scene(h: int = 300, w: int = 400) -> np.ndarray:
    rng = np.random.default_rng(3)
    img = np.zeros((h, w, 3), np.int16)
    for i, band in enumerate(PLAN_BANDS):
        img[i * h // 4 : (i + 1) * h // 4] = band
    img += rng.integers(-3, 4, img.shape, dtype=np.int16)
    return np.clip(img, 0, 255).astype(np.uint8)


def plan_params(**kw) -> Params:
    base = dict(doc_mode=DOC_PLAN, palette=WATERCOLOR_STARTER)
    base.update(kw)
    return Params(**base)


def run_plan(img=None, **kw):
    pipe = Pipeline()
    pipe.set_source(LoadedImage(rgb=img if img is not None else make_plan_scene(),
                                source_path="synthetic"))
    p = plan_params(**kw)
    return pipe, p, pipe.run(p)


# A light disc on a dark ground: the one contour is a circle, so a course that
# follows the drawing must run tangential to it. Horizontal bands could not tell
# andamento apart from a plain lattice.
BLOB_R = 90.0


def make_blob_scene(h: int = 300, w: int = 300) -> np.ndarray:
    yy, xx = np.mgrid[0:h, 0:w]
    inside = ((xx - w / 2.0) ** 2 + (yy - h / 2.0) ** 2) <= BLOB_R ** 2
    img = np.where(inside[..., None], np.array([226, 220, 208]), np.array([54, 58, 66]))
    rng = np.random.default_rng(5)
    return np.clip(img + rng.integers(-3, 4, img.shape), 0, 255).astype(np.uint8)


def painted_fraction(tesserae, grout: float, n: int = 320) -> float:
    """Share of the picture the drawn stones actually cover, by rasterizing them."""
    acc = np.zeros((n, n), dtype=bool)
    yy, xx = np.mgrid[0:n, 0:n]
    for te in tesserae:
        mx = sum(p[0] for p in te.points) / 4.0
        my = sum(p[1] for p in te.points) / 4.0
        px = np.array([(mx + (p[0] - mx) * (1 - grout)) * n for p in te.points])
        py = np.array([(my + (p[1] - my) * (1 - grout)) * n for p in te.points])
        x0, x1 = int(max(0, px.min())), int(min(n, px.max() + 1))
        y0, y1 = int(max(0, py.min())), int(min(n, py.max() + 1))
        if x0 >= x1 or y0 >= y1:
            continue
        sx, sy = xx[y0:y1, x0:x1], yy[y0:y1, x0:x1]
        inside = np.ones(sx.shape, dtype=bool)
        for i in range(4):
            ax, ay = px[i], py[i]
            bx, by = px[(i + 1) % 4], py[(i + 1) % 4]
            inside &= ((bx - ax) * (sy - ay) - (by - ay) * (sx - ax)) >= -1e-9
        acc[y0:y1, x0:x1] |= inside
    return float(acc.mean())


def mosaic_params(**kw) -> Params:
    base = dict(doc_mode=DOC_MOSAIC, palette=WATERCOLOR_STARTER, tiles_x=16)
    base.update(kw)
    return Params(**base)


def run_lattice(**kw):
    return run_mosaic(tessera_layout="grid", **kw)


def run_mosaic(img=None, **kw):
    pipe = Pipeline()
    pipe.set_source(LoadedImage(rgb=img if img is not None else make_plan_scene(),
                                source_path="synthetic"))
    p = mosaic_params(**kw)
    return pipe, p, pipe.run(p)


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
    from studio.ui.pdf_print import write_scale_pdf

    p = Params(manual_targets=(Target(hex="#7a5c3a"),), card_w=11.0)
    res = Pipeline().run(p)
    check(res.scene.inches_per_px > 0, "card width sets the physical scale")
    svg = scene_to_svg(res.scene)
    check('width="11in"' in svg, "SVG carries the physical width (11in)")
    p_cm = p.copy(card_w=28.0, canvas_units="cm")
    svg_cm = scene_to_svg(Pipeline().run(p_cm).scene)
    check('width="28cm"' in svg_cm, "cm units label the SVG in centimetres")
    with tempfile.TemporaryDirectory() as td:
        pdf_path = os.path.join(td, "board.pdf")
        w_in, h_in = write_scale_pdf(res.scene, pdf_path)
        check(abs(w_in - 11.0) < 1e-6, f"the page is exactly the card width ({w_in:.3f} in)")
        with open(pdf_path, "rb") as fh:
            check(fh.read(5) == b"%PDF-", "PDF export starts with %PDF")
        import fitz
        doc = fitz.open(pdf_path)
        try:
            check(doc.page_count == 1, "a card-sized board is a single page")
            check(abs(doc[0].rect.width / 72.0 - 11.0) < 0.02,
                  "the PDF page really measures 11 in across")
            # The whole point of the vector path: text survives as text.
            check(len(doc[0].get_text().strip()) > 40,
                  "board text is real text in the PDF, not a raster")
        finally:
            doc.close()
    try:
        write_scale_pdf(Pipeline().run(Params()).scene, os.devnull)
        check(False, "an unscaled scene is refused a to-scale PDF")
    except ValueError:
        check(True, "an unscaled scene is refused a to-scale PDF")


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
    from studio.ui.pdf_print import write_tiled_pdf

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

    # --- painting plan through the real window --------------------------------
    plan_p = Params(doc_mode=DOC_PLAN, palette=WATERCOLOR_STARTER, wash_tiers=5,
                    plan_colors=9, min_passage_pct=2.5, paper_w=15.0, paper_h=11.0,
                    show_plan_tips=False)
    plan_panel = ControlsPanel(plan_p)
    plan_panel.apply(plan_p)
    check(plan_panel.collect(plan_p) == plan_p, "controls round-trip the plan fields")
    check(not plan_panel.medium.isEnabled(), "plan mode locks the medium to watercolor")
    check(not plan_panel._g_plan.isHidden(), "plan mode shows the plan controls")
    check(plan_panel._g_ladder.isHidden() and plan_panel._g_wheel.isHidden()
          and plan_panel._g_board.isHidden(), "plan mode hides the board-only controls")
    plan_panel.apply(Params())
    check(plan_panel.medium.isEnabled(), "board mode unlocks the medium again")
    check(plan_panel._g_plan.isHidden(), "board mode hides the plan controls")
    check(not plan_panel._g_board.isHidden(), "board mode shows the board controls")

    # Switch the live window into plan mode via the panel, as a user would.
    win.params = win.params.copy(palette=WATERCOLOR_STARTER)
    win.controls.apply(win.params.copy(doc_mode=DOC_PLAN))
    win._on_controls_changed()
    win._render_now()
    deadline = time.monotonic() + 30.0
    while (win._busy or win._pending or win._timer.isActive()
           or win._result is None or win._result.passage_map is None) and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(0.01)
    check(win.params.doc_mode == DOC_PLAN, "the window switched to plan mode")
    check(win._result is not None and win._result.passage_map is not None,
          "the window renders a painting plan")
    check(len(win._result.passes) >= 1, "the rendered plan has washes")
    check(len(win._result.layout.pass_rects) == len(win._result.passes),
          "the plan layout exposes a rect per wash")
    # Clicking a wash row highlights it; clicking again clears it.
    (prx, pry, prw, prh), pidx = win._result.layout.pass_rects[0]
    win._on_preview_click(prx + prw / 2, pry + prh / 2, 0)
    check(win.params.selected_pass == pidx, "clicking a wash selects it")
    win._on_preview_click(prx + prw / 2, pry + prh / 2, 0)
    check(win.params.selected_pass == -1, "clicking the same wash again clears the highlight")
    check(win._doc_stem() == "painting-plan", "exports are named for the plan")

    # --- mosaic guide through the real window ---------------------------------
    mos_p = Params(doc_mode=DOC_MOSAIC, palette=WATERCOLOR_STARTER, tiles_x=18,
                   mosaic_colors=9, tessera_mm=15.0, grout_frac=0.2,
                   tessera_jitter=0.4, show_mosaic_legend=False)
    mos_panel = ControlsPanel(mos_p)
    mos_panel.apply(mos_p)
    check(mos_panel.collect(mos_p) == mos_p, "controls round-trip the mosaic fields")
    check(not mos_panel._g_mosaic.isHidden(), "mosaic mode shows the mosaic controls")
    check(mos_panel._g_plan.isHidden() and mos_panel._g_board.isHidden(),
          "mosaic mode hides the other documents' controls")
    check(not mos_panel.medium.isEnabled(), "mosaic mode locks the medium to watercolor")

    win.controls.apply(win.params.copy(doc_mode=DOC_MOSAIC, tiles_x=18))
    win._on_controls_changed()
    win._render_now()
    deadline = time.monotonic() + 30.0
    while (win._busy or win._pending or win._timer.isActive()
           or win._result is None or win._result.mosaic is None) and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(0.01)
    check(win.params.doc_mode == DOC_MOSAIC, "the window switched to mosaic mode")
    check(win._result is not None and win._result.mosaic is not None,
          "the window renders a mosaic guide")
    check(win.act_grid.isVisible() and win.act_grid.isEnabled(),
          "the to-scale grid export is offered in mosaic mode")
    check(win._doc_stem() == "mosaic-guide", "exports are named for the mosaic")
    # Clicking a legend row highlights that colour, and clicking again clears it.
    if win._result.layout.color_rects:
        (crx, cry, crw, crh), cidx = win._result.layout.color_rects[0]
        win._on_preview_click(crx + crw / 2, cry + crh / 2, 0)
        check(win.params.selected_color == cidx, "clicking a legend row selects that colour")
        win._on_preview_click(crx + crw / 2, cry + crh / 2, 0)
        check(win.params.selected_color == -1, "clicking it again clears the highlight")
    # The to-scale grid really writes a multi-page vector PDF.
    sheet = build_grid_sheet(win.params, win._result.mosaic)
    with tempfile.TemporaryDirectory() as td:
        out = os.path.join(td, "grid.pdf")
        pages, prows, pcols = write_tiled_pdf(sheet, out)
        check(os.path.getsize(out) > 2000, f"the tiled grid PDF is written ({pages} pages)")
        check(pages == prows * pcols, "the PDF has one page per tile")
        import fitz
        doc = fitz.open(out)
        try:
            check(doc.page_count == pages, "the PDF really holds that many pages")
            a4_w = PAPER_SIZES_IN["A4"][0] * 72.0
            check(abs(doc[0].rect.width - a4_w) < 1.5, "pages come out A4")
            check(len(doc[0].get_text().strip()) > 0, "cell numbers are real text, not a raster")
        finally:
            doc.close()

    win._thread.quit()
    win._thread.wait(3000)
    QSettings().clear()


def group18_plan_segmentation():
    print("\n[18] Painting plan — value tiers & passages")
    _pipe, _p, res = run_plan()
    pm = res.passage_map
    check(pm is not None, "plan mode produces a passage map")
    check(len(pm.tier_fracs) == 4, f"wash_tiers tiers reported (got {len(pm.tier_fracs)})")
    check(abs(sum(pm.tier_fracs) - 1.0) < 1e-6, "tier area fractions sum to 1")
    check(all(pm.tier_centers[i] >= pm.tier_centers[i + 1] for i in range(len(pm.tier_centers) - 1)),
          "tier centers run lightest -> darkest")
    # Four flat bands must not shatter into dozens of passages.
    check(4 <= len(pm.passages) <= 8,
          f"four flat bands give a handful of passages (got {len(pm.passages)})")
    check(abs(sum(p.area_frac for p in pm.passages) - 1.0) < 1e-6,
          "passage areas cover the whole image")
    check([p.id for p in pm.passages] == list(range(1, len(pm.passages) + 1)),
          "passages are numbered 1..n in painting order")
    ls = [p.lstar for p in pm.passages]
    check(all(pm.passages[i].tier <= pm.passages[i + 1].tier for i in range(len(ls) - 1)),
          "passage numbering follows the value tiers")
    # The near-white band is reserved; the dark band never is.
    top = max(pm.passages, key=lambda p: p.lstar)
    bottom = min(pm.passages, key=lambda p: p.lstar)
    check(top.reserved, f"the near-white band is reserved (L* {top.lstar:.0f})")
    check(not bottom.reserved, f"the dark band is painted, not reserved (L* {bottom.lstar:.0f})")
    # Every label indexes a real passage, and each passage owns some pixels.
    check(int(pm.labels.min()) == 0 and int(pm.labels.max()) == len(pm.passages) - 1,
          "label image indexes exactly the passage list")
    check(len(np.unique(pm.labels)) == len(pm.passages), "every passage owns pixels")
    # Centroid labels land inside their own shape (concave shapes included).
    h, w = pm.labels.shape
    inside = all(
        pm.labels[min(h - 1, int(p.cy * h)), min(w - 1, int(p.cx * w))] == i
        for i, p in enumerate(pm.passages)
    )
    check(inside, "each passage's number sits inside its own shape")
    # Adjacency is symmetric.
    sym = all(
        a in pm.adjacency.get(b, {}) and pm.adjacency[b][a] == n
        for a, nbrs in pm.adjacency.items() for b, n in nbrs.items()
    )
    check(sym, "passage adjacency is symmetric")
    # Determinism, and a coarser threshold gives no more shapes.
    _p2, _q2, res2 = run_plan()
    check([p.target_hex for p in res2.passage_map.passages]
          == [p.target_hex for p in pm.passages], "segmentation is deterministic")
    _p3, _q3, res3 = run_plan(min_passage_pct=6.0)
    check(len(res3.passage_map.passages) <= len(pm.passages),
          "a coarser smallest-shape setting never adds passages")
    check(len(res.passage_map.passages) <= plan_regions.MAX_PASSAGES,
          "passage count is capped")


def group19_plan_schedule():
    print("\n[19] Painting plan — wash schedule")
    _pipe, p, res = run_plan()
    passes = res.passes
    check(len(passes) >= 2, f"the schedule has several washes (got {len(passes)})")
    check([w.index for w in passes] == list(range(1, len(passes) + 1)),
          "washes are numbered 1..n")
    # The load-bearing invariant: no wash ever goes down over a darker one.
    by_id = {p_.id: p_ for p_ in res.passage_map.passages}

    def mean_l(w):
        tot = sum(by_id[i].area_frac for i in w.passage_ids)
        return sum(by_id[i].lstar * by_id[i].area_frac for i in w.passage_ids) / tot

    lums = [mean_l(w) for w in passes]
    check(all(lums[i] >= lums[i + 1] - 1e-9 for i in range(len(lums) - 1)),
          f"washes run strictly light -> dark ({[f'{v:.0f}' for v in lums]})")
    # Reserved shapes are never scheduled; painted ones are scheduled once.
    scheduled = [i for w in passes for i in w.passage_ids]
    reserved = {p_.id for p_ in res.passage_map.passages if p_.reserved}
    check(not (set(scheduled) & reserved), "reserved whites are never painted")
    check(len(scheduled) == len(set(scheduled)), "each passage is painted in exactly one wash")
    check(set(scheduled) | reserved == {p_.id for p_ in res.passage_map.passages},
          "every passage is either painted or reserved")
    check(all(w.recipe.dilution > 0 for w in passes), "every wash carries a dilution")
    check(all(w.volume_ml >= plan_schedule.MIN_ML for w in passes),
          "every wash gets a mixable volume")
    check(all("brush-load" in w.volume_text for w in passes), "volumes speak in brush-loads")
    check(all(w.why and w.tip for w in passes), "every wash explains itself and teaches something")
    check(len({w.tip for w in passes}) == len(passes), "wash tips do not repeat")
    check(passes[-1].recipe.dilution >= passes[0].recipe.dilution,
          "the darks are mixed stronger than the underwash")

    # Parts arithmetic: whole numbers that track the solved weights.
    check(plan_schedule.mix_parts([1.0]) == (1,), "a single paint is one part")
    check(plan_schedule.mix_parts([0.5, 0.5]) == (1, 1), "an even mix is 1:1")
    check(plan_schedule.mix_parts([0.75, 0.25]) == (3, 1), "0.75/0.25 reads as 3:1")
    check(plan_schedule.mix_parts([0.66, 0.34]) == (2, 1), "0.66/0.34 reads as 2:1")
    for w in passes:
        parts = plan_schedule.mix_parts([s.weight for s in w.recipe.steps])
        check(len(parts) == len(w.recipe.steps) and all(v >= 1 for v in parts),
              f"wash {w.index} parts are whole and complete ({parts})")
        biggest = max(range(len(w.recipe.steps)), key=lambda i: w.recipe.steps[i].weight)
        check(parts[biggest] == max(parts),
              f"wash {w.index} gives the dominant paint the most parts")

    # Volume tracks paper area, and only paper area.
    area = plan_schedule.paper_area_cm2(10.0, 7.0, "in")
    check(abs(area - 10 * 7 * 2.54 ** 2) < 1e-6, "paper area converts inches to cm²")
    check(abs(plan_schedule.paper_area_cm2(20.0, 30.0, "cm") - 600.0) < 1e-6,
          "paper area in cm is used as-is")
    small = plan_schedule.wash_volume_ml(0.5, area, False)
    big = plan_schedule.wash_volume_ml(0.5, area * 4, False)
    check(big > small, "a bigger sheet needs more wash")
    check(plan_schedule.wash_volume_ml(0.5, area, True)
          > plan_schedule.wash_volume_ml(0.5, area, False),
          "wetting the paper first costs extra volume")
    check(plan_schedule.wash_volume_ml(0.0001, area, False) == plan_schedule.MIN_ML,
          "a tiny shape still gets a mixable minimum")
    check(plan_schedule.wash_volume_ml(0.5, 0.0, False) == 0.0,
          "no paper size means no invented volume")
    check("set a paper size" in plan_schedule.volume_text(0.0),
          "a missing paper size says so instead of printing 0 ml")

    # Edge language reflects what is actually adjacent.
    joined = " ".join(w.edges for w in passes)
    check("already dry" in joined, "later washes are told which neighbours are dry")
    if reserved:
        check("bare paper" in joined, "washes touching a reserved white are warned off it")


def group20_plan_document():
    print("\n[20] Painting plan — document")
    _pipe, p, res = run_plan()
    scene = res.scene
    names = [ly.name for ly in scene.layers]
    check(names == list(PLAN_LAYERS), f"plan layers present in order (got {names})")
    check(scene.height > 600, "the plan is a full document")
    # Reference photo + big map + one mini-map per wash.
    check(len(scene.patches) == 2 + len(res.passes),
          f"photo, map and one mini-map per wash (got {len(scene.patches)})")
    check(res.layout.photo_rect is not None, "the plan exposes a photo rect for the eyedropper")
    check(len(res.layout.pass_rects) == len(res.passes), "every wash is clickable")
    rects = [r for r, _i in res.layout.pass_rects]
    check(all(rects[i][1] < rects[i + 1][1] for i in range(len(rects) - 1)),
          "wash rows do not overlap")
    # Section toggles are compose-only.
    _pp, _qq, res_off = run_plan(show_plan_tips=False)
    check(res_off.scene.height < scene.height, "turning off a section shortens the plan")
    check(res_off.passes == res.passes, "a section toggle does not disturb the schedule")
    # Selecting a wash highlights it without changing the plan.
    _ps, _qs, res_sel = run_plan(selected_pass=0)
    sel_layer = res_sel.scene.layer(LAYER_PLAN_PASSES)
    base_layer = scene.layer(LAYER_PLAN_PASSES)
    check(len(sel_layer.strokes) == len(base_layer.strokes) + 1,
          "selecting a wash draws exactly one highlight")
    check(res_sel.passes == res.passes, "selection never changes the schedule")
    # Generated sentences stay inside the page.
    for ly in res.scene.layers:
        for t in ly.texts:
            width = len(t.text) * t.size * 0.62
            end = t.x + (width if t.anchor == "start" else width / 2.0)
            check(end <= scene.width + 1, f"text stays on the page ({t.text[:34]!r})")
            break   # one representative line per layer keeps the log readable
    longest = max(
        (t for ly in res.scene.layers for t in ly.texts),
        key=lambda t: len(t.text) * t.size,
    )
    check(longest.x + len(longest.text) * longest.size * 0.62 <= scene.width + 1,
          f"the longest generated line is clipped to the page ({len(longest.text)} chars)")
    # SVG carries the map rasters and the Mixbox attribution.
    svg = scene_to_svg(scene)
    check(svg.count("<image") == len(scene.patches), "every map raster is embedded in the SVG")
    check("CC BY-NC" in svg, "the plan carries the Mixbox attribution")
    for name in PLAN_LAYERS:
        check(f'inkscape:label="{name}"' in svg, f"SVG has the {name} layer")
    # With no photo the plan says what to do instead of failing.
    empty = Pipeline().run(plan_params())
    check(empty.passage_map is None and empty.passes == (), "no photo means no plan")
    head = " ".join(t.text for t in empty.scene.layer("header").texts)
    check("reference photo" in head, "the empty plan asks for a photo")


def group21_plan_cache():
    print("\n[21] Painting plan — caching & mode switching")
    pipe, p, _res = run_plan()
    check(pipe.run(p).recomputed == [], "a repeat plan run recomputes nothing")
    # Paper size only re-costs the washes.
    p2 = p.copy(paper_h=14.0)
    check(pipe.run(p2).recomputed == ["schedule"], "paper size only re-runs the schedule")
    v_small = pipe.run(p).passes[0].volume_ml
    v_big = pipe.run(p2).passes[0].volume_ml
    check(v_big > v_small, "a bigger sheet asks for more wash")
    # Segmentation settings re-segment and everything below.
    check(pipe.run(p.copy(plan_colors=5)).recomputed == ["passages", "solve", "schedule"],
          "colour-area count re-segments")
    check(pipe.run(p.copy(wash_tiers=5)).recomputed == ["passages", "solve", "schedule"],
          "tier count re-segments")
    # Renaming a paint is still free, and still reaches the printed plan.
    pipe.run(p)                     # back to the baseline settings
    first = _res.passes[0].recipe.steps[0]
    renamed = tuple(
        Paint(name="Renamed Ochre", hex=q.hex, is_white=q.is_white, opacity=q.opacity,
              staining=q.staining, granulating=q.granulating)
        if q.hex == first.paint_hex else q
        for q in p.palette
    )
    res_r = pipe.run(p.copy(palette=renamed))
    check(res_r.recomputed == [], "renaming a paint recomputes nothing in plan mode")
    printed = " ".join(t.text for t in res_r.scene.layer(LAYER_PLAN_PASSES).texts)
    check("Renamed Ochre" in printed, "a renamed paint still reaches the printed washes")
    check(pipe.run(p.copy(show_plan_map=False)).recomputed == [], "section toggles are compose-only")
    # Board mode never reports plan stages, and vice versa.
    board = pipe.run(p.copy(doc_mode=DOC_BOARD))
    check("passages" not in board.recomputed and "schedule" not in board.recomputed,
          f"board mode never mentions plan stages (got {board.recomputed})")
    check("ladders" not in pipe.run(p).recomputed, "plan mode never mentions ladders")
    # Regression: a plan run must not poison the board's photo extraction.
    check(len(board.recipes) > 0 and len(board.auto_targets) > 0,
          "board mode still extracts photo targets after a plan run")
    plan_again = pipe.run(p)
    check(plan_again.passage_map is not None and len(plan_again.passes) == len(_res.passes),
          "switching back to the plan restores it unchanged")


def group22_mosaic_grid():
    print("\n[22] Mosaic — tesserae & colour stock")
    _pipe, p, res = run_lattice()
    g = res.mosaic
    check(g is not None, "mosaic mode produces a grid")
    check(g.layout == "grid", "the lattice layout is honoured")
    check(g.cols == 16, f"tiles_x sets the column count (got {g.cols})")
    # 300x400 source -> rows track the aspect.
    src_h, src_w = make_plan_scene().shape[:2]
    check(g.rows == round(g.cols * src_h / src_w),
          f"rows follow the photo's aspect (got {g.rows})")
    check(g.cells.shape == (g.rows, g.cols), "the cell map is rows x cols")
    check(g.total == g.rows * g.cols, "the lattice fills every cell")
    check(sum(g.counts) == g.total, "the per-colour counts cover every tessera")
    check(len(g.colors) == len(g.counts), "every colour has a count")
    check(all(c > 0 for c in g.counts), "no colour is left unused")
    check(int(g.cells.min()) == 0 and int(g.cells.max()) == len(g.colors) - 1,
          "cell values index exactly the colour list")
    check(abs(sum(g.fraction(i) for i in range(len(g.colors))) - 1.0) < 1e-9,
          "colour fractions sum to 1")
    # Lightest first — which is also the painting order.
    ls = [color.lstar(color.hex_to_rgb(c)) for c in g.colors]
    check(all(ls[i] >= ls[i + 1] for i in range(len(ls) - 1)),
          f"colours are numbered lightest first ({[f'{v:.0f}' for v in ls]})")
    # Determinism, and the colour count is honoured.
    _p2, _q2, res2 = run_lattice()
    check(res2.mosaic.colors == g.colors, "quantization is deterministic")
    check(np.array_equal(res2.mosaic.cells, g.cells), "the cell map is deterministic")
    _p3, _q3, res3 = run_lattice(mosaic_colors=5)
    check(len(res3.mosaic.colors) == 5, "mosaic_colors sets the palette size")
    # Bounds are clamped rather than trusted.
    check(mosaic_grid.grid_shape(400, 300, 500)[0] == mosaic_grid.MAX_TILES,
          "an absurd tiles_x clamps to MAX_TILES")
    check(mosaic_grid.grid_shape(400, 300, 1)[0] == mosaic_grid.MIN_TILES,
          "a tiny tiles_x clamps to MIN_TILES")
    # Recipes are index-aligned with the colours.
    check(len(res.recipes) >= len(g.colors), "every mosaic colour got a recipe")
    check(all(res.recipes[i].target_hex == c for i, c in enumerate(g.colors)),
          "recipes stay index-aligned with the colour stock")


def group23_mosaic_tesserae():
    print("\n[23] Mosaic — tesserae geometry")
    _pipe, p, res = run_lattice()
    g = res.mosaic
    colors = mosaic_doc.cell_colors(g, res.recipes)
    check(len(colors) == len(g.colors), "every colour maps to a painted colour")
    check(colors[0] == res.recipes[0].mixed_hex,
          "tesserae are drawn in the colour the palette actually reaches")
    W, H = 640.0, 640.0 * g.rows / g.cols
    strokes = mosaic_doc.tesserae_strokes(g, colors, 0.0, 0.0, W, H,
                                          p.grout_frac, p.tessera_jitter, p.background)
    check(len(strokes) == g.total, "one quad per tessera")
    check(all(len(s.points) == 4 and s.closed for s in strokes), "each tessera is a closed quad")
    # The invariant the whole jitter model rests on: a corner can never leave
    # its own footprint, so two tesserae can never touch or overlap.
    cw, ch = W / g.cols, H / g.rows
    escaped = 0
    for gf in (0.0, 0.05, 0.18, 0.30):
        for jt in (0.0, 0.5, 1.0, 2.0):   # 2.0 is out of range and must clamp
            st = mosaic_doc.tesserae_strokes(g, colors, 0.0, 0.0, W, H, gf, jt, p.background)
            for i, s in enumerate(st):
                r, c = divmod(i, g.cols)
                for px, py in s.points:
                    if not (c * cw - 1e-9 <= px <= (c + 1) * cw + 1e-9
                            and r * ch - 1e-9 <= py <= (r + 1) * ch + 1e-9):
                        escaped += 1
    check(escaped == 0, f"no tessera corner ever leaves its cell ({escaped} escapes)")
    # Zero jitter is a perfect grid; more jitter genuinely moves things.
    flat = mosaic_doc.tesserae_strokes(g, colors, 0.0, 0.0, W, H, 0.18, 0.0, p.background)
    rough = mosaic_doc.tesserae_strokes(g, colors, 0.0, 0.0, W, H, 0.18, 1.0, p.background)
    check(all(len({round(x, 6) for x, _y in s.points}) == 2 for s in flat[:20]),
          "zero jitter gives axis-aligned squares")
    moved = sum(1 for a, b in zip(flat, rough) if a.points != b.points)
    check(moved == len(flat), "full jitter displaces every tessera")
    same = mosaic_doc.tesserae_strokes(g, colors, 0.0, 0.0, W, H, 0.18, 1.0, p.background)
    check([s.points for s in same] == [s.points for s in rough],
          "the jitter pattern is stable across renders")
    # The grout floor is a watercolor requirement, not a style choice.
    check(mosaic_doc.draw_grout(0.0) == mosaic_flow.MIN_GROUT,
          "grout is floored so two wet squares can never touch")
    check(mosaic_doc.draw_grout(0.25) == 0.25, "a generous grout is used as given")
    # Row labels run past Z without repeating.
    labels = [mosaic_doc.row_label(i) for i in range(60)]
    check(labels[0] == "A" and labels[25] == "Z" and labels[26] == "AA",
          f"row labels go A..Z then AA (got {labels[25]}, {labels[26]})")
    check(len(set(labels)) == len(labels), "row labels never repeat")


def group24_mosaic_document():
    print("\n[24] Mosaic — document & to-scale sheet")
    _pipe, p, res = run_mosaic()
    scene = res.scene
    names = [ly.name for ly in scene.layers]
    check(names == list(MOSAIC_LAYERS), f"mosaic layers present in order (got {names})")
    check(len(res.layout.color_rects) == len(res.mosaic.colors), "every colour row is clickable")
    check(res.layout.photo_rect is not None, "the mosaic exposes a photo rect")
    # Section toggles are compose-only.
    _p2, _q2, off = run_mosaic(show_mosaic_grid=False)
    check(off.scene.height < scene.height, "turning off the grid shortens the document")
    check(off.mosaic.colors == res.mosaic.colors, "a section toggle leaves the grid alone")
    # Selecting a colour highlights without changing anything.
    _p3, _q3, sel = run_mosaic(selected_color=0)
    check(sel.mosaic.colors == res.mosaic.colors, "selection never changes the colours")
    check(len(sel.scene.layer(LAYER_MOSAIC_LEGEND).strokes)
          == len(scene.layer(LAYER_MOSAIC_LEGEND).strokes) + 1,
          "selecting a colour draws exactly one highlight")
    # Text stays on the page.
    longest = max((t for ly in scene.layers for t in ly.texts),
                  key=lambda t: len(t.text) * t.size)
    check(longest.x + len(longest.text) * longest.size * 0.62 <= scene.width + 1,
          f"the longest generated line fits the page ({len(longest.text)} chars)")
    # SVG export.
    svg = scene_to_svg(scene)
    check("CC BY-NC" in svg, "the mosaic guide carries the Mixbox attribution")
    for name in MOSAIC_LAYERS:
        check(f'inkscape:label="{name}"' in svg, f"SVG has the {name} layer")
    # No photo -> a message, not a crash.
    empty = Pipeline().run(mosaic_params())
    check(empty.mosaic is None, "no photo means no grid")
    head = " ".join(t.text for t in empty.scene.layer("header").texts)
    check("reference photo" in head, "the empty mosaic asks for a photo")

    # The to-scale sheet carries a real physical scale.
    sheet = mosaic_doc.build_grid_sheet(p, res.mosaic)
    ipp = sheet.inches_per_px
    check(ipp > 0, "the grid sheet carries a physical scale")
    cell_in = mosaic_doc.SHEET_CELL * ipp
    check(abs(cell_in - p.tessera_mm / 25.4) < 1e-9,
          f"one sheet cell prints as one tessera ({cell_in * 25.4:.2f} mm)")
    grid_in = res.mosaic.cols * cell_in
    check(sheet.width * ipp > grid_in, "the sheet is at least as wide as the grid")
    sheet2 = mosaic_doc.build_grid_sheet(p.copy(tessera_mm=p.tessera_mm * 2), res.mosaic)
    check(abs(sheet2.inches_per_px - 2 * ipp) < 1e-12,
          "doubling the tessera size doubles the printed scale")


def group25_mosaic_tiling():
    print("\n[25] Mosaic — multi-page tile plan")
    a4_w, a4_h = PAPER_SIZES_IN["A4"]
    margin, overlap = 10.0 / 25.4, 10.0 / 25.4
    pw, ph = a4_w - 2 * margin, a4_h - 2 * margin
    # A sheet that fits is one clean page.
    tiles, rows, cols = plan_tiles(100, 100, 0, 0.05, pw, ph, overlap)
    check((rows, cols) == (1, 1) and len(tiles) == 1, "a sheet that fits gives one page")
    check(tiles[0] == (0, 0, 0.0, 0.0), "the single tile starts at the origin")
    # A 60 cm grid needs several sheets, and the stride respects the overlap.
    ipp = 0.05
    wide = int(round((60.0 / 2.54) / ipp))
    tiles, rows, cols = plan_tiles(wide, wide, 0, ipp, pw, ph, overlap)
    check(cols == math.ceil((60.0 / 2.54 - overlap) / (pw - overlap)),
          f"a 60 cm grid tiles across the expected column count (got {cols})")
    check(len(tiles) == rows * cols, "the plan emits one tile per page")
    check(all(t[0] < rows and t[1] < cols for t in tiles), "tile indices stay in range")
    stride = (tiles[1][2] - tiles[0][2]) * ipp
    check(abs(stride - (pw - overlap)) < 1e-9,
          f"adjacent tiles overlap by exactly the tape band ({stride:.3f} in)")
    check(plan_tiles(wide, wide, 0, ipp, pw, ph, 0.0)[2] <= cols,
          "no overlap needs no more pages than overlapping does")
    # Bad inputs are refused rather than producing a silent mess.
    for bad, why in (
        (lambda: plan_tiles(100, 100, 0, 0.0, pw, ph, overlap), "no physical scale"),
        (lambda: plan_tiles(100, 100, 0, 0.05, overlap, ph, overlap), "overlap >= paper"),
    ):
        try:
            bad()
            check(False, f"plan_tiles rejects {why}")
        except ValueError:
            check(True, f"plan_tiles rejects {why}")


def group26_mosaic_cache():
    print("\n[26] Mosaic — caching & mode switching")
    pipe, p, res = run_mosaic()
    check(pipe.run(p).recomputed == [], "a repeat mosaic run recomputes nothing")
    # Drawing decisions must never reach the cache.
    for field, value in (("grout_frac", 0.05), ("tessera_jitter", 0.2),
                         ("tessera_mm", 25.0), ("show_mosaic_legend", False),
                         ("selected_color", 1)):
        check(pipe.run(p.copy(**{field: value})).recomputed == [],
              f"{field} is compose-only")
    # Grid settings re-quantize and re-solve.
    check(pipe.run(p.copy(tiles_x=20)).recomputed == ["mosaic", "solve"],
          "tiles_x re-lays the tesserae")
    check(pipe.run(p.copy(mosaic_colors=6)).recomputed == ["mosaic", "solve"],
          "mosaic_colors re-quantizes")
    pipe.run(p)
    # Renaming a paint is free, and still reaches the printed legend.
    first = res.recipes[0].steps[0]
    renamed = tuple(
        Paint(name="Renamed Sienna", hex=q.hex, is_white=q.is_white, opacity=q.opacity,
              staining=q.staining, granulating=q.granulating)
        if q.hex == first.paint_hex else q
        for q in p.palette
    )
    res_r = pipe.run(p.copy(palette=renamed))
    check(res_r.recomputed == [], "renaming a paint recomputes nothing in mosaic mode")
    printed = " ".join(t.text for t in res_r.scene.layer(LAYER_MOSAIC_LEGEND).texts)
    check("Renamed Sienna" in printed, "a renamed paint still reaches the printed legend")
    # Documents stay out of each other's stages.
    board = pipe.run(p.copy(doc_mode=DOC_BOARD))
    check("mosaic" not in board.recomputed,
          f"board mode never mentions the mosaic stage (got {board.recomputed})")
    check(len(board.recipes) > 0 and len(board.auto_targets) > 0,
          "board mode still extracts photo targets after a mosaic run")
    mos = pipe.run(p)
    check("ladders" not in mos.recomputed and "passages" not in mos.recomputed,
          f"mosaic mode never mentions plan stages (got {mos.recomputed})")
    check(mos.mosaic is not None and mos.mosaic.colors == res.mosaic.colors,
          "switching back to the mosaic restores it unchanged")
    plan = pipe.run(p.copy(doc_mode=DOC_PLAN))
    check(plan.passage_map is not None and plan.mosaic is None,
          "plan mode after a mosaic run builds passages, not tesserae")


def group27_mosaic_andamento():
    print("\n[27] Mosaic — andamento (courses that follow the drawing)")
    _pipe, p, res = run_mosaic()          # default layout is "flow"
    g = res.mosaic
    check(g.layout == "flow", "flow is the default layout")
    check(g.cells is None, "the flow layout has no lattice to address")
    check(g.total > 0, f"the flow layout lays stones ({g.total})")
    check(sum(g.counts) == g.total, "counts cover every stone laid")
    check(all(0 <= t.color < len(g.colors) for t in g.tesserae),
          "every stone takes a colour from the stock")
    check(all(len(t.points) == 4 for t in g.tesserae), "every stone is a quad")
    check(all(0.0 <= v <= 1.0 for t in g.tesserae for pt in t.points for v in pt),
          "stones stay inside the picture")
    # The mosaic must cover the whole picture. Bare patches larger than the
    # grout read as deliberate white shapes, not as a mosaic, so the filling
    # passes cut smaller stones until the gaps close.
    covered = painted_fraction(g.tesserae, mosaic_doc.draw_grout(p.grout_frac))
    check(covered > 0.60, f"the stones cover the picture ({covered * 100:.0f}% painted)")
    check(g.total > g.cols * g.rows * 0.85,
          f"filling passes lay more stones than a plain lattice would ({g.total})")
    edges = sorted({round(math.dist(t.points[0], t.points[1]), 4) for t in g.tesserae})
    check(len(edges) > 1, f"stones come in several sizes ({len(edges)} distinct)")
    check(edges[0] < edges[-1] * 0.6, "the smallest fillers are genuinely cut down")
    # Without the small fillers the same picture is left with holes.
    _pn, _qn, none_res = run_mosaic(tiles_x=16)
    check(painted_fraction(none_res.mosaic.tesserae, 0.10) > 0.55,
          "coverage holds at other stone sizes too")
    # Determinism.
    _p2, _q2, res2 = run_mosaic()
    check([t.points for t in res2.mosaic.tesserae] == [t.points for t in g.tesserae],
          "the flow layout is deterministic")

    # --- what makes it a mosaic rather than a grid --------------------------
    def angle_of(t):
        (x0, y0), (x1, y1) = t.points[0], t.points[1]
        return math.atan2(y1 - y0, x1 - x0) % (math.pi / 2.0)

    # The real test: on a disc, a course that follows the drawing must run
    # tangential to the circle. Bands of colour could not distinguish andamento
    # from a lattice, so this uses a curved contour on purpose.
    _pb, _qb, blob = run_mosaic(img=make_blob_scene(), tiles_x=24)
    bg = blob.mosaic
    ring = []
    for t in bg.tesserae:
        dx, dy = t.cx - 0.5, t.cy - 0.5
        r = math.hypot(dx, dy)
        if 0.20 < r < 0.34:                      # the courses hugging the rim
            radial = math.atan2(dy, dx)
            tangent = (radial + math.pi / 2.0) % (math.pi / 2.0)
            err = abs(angle_of(t) - tangent) % (math.pi / 2.0)
            ring.append(min(err, math.pi / 2.0 - err))
    check(len(ring) > 20, f"the disc has a rim course to measure ({len(ring)} stones)")
    aligned = sum(1 for e in ring if e < 0.35)   # within ~20 degrees of tangent
    check(aligned > 0.7 * len(ring),
          f"rim stones run tangential to the contour ({aligned}/{len(ring)})")
    spread = float(np.std([angle_of(t) for t in bg.tesserae]))
    check(spread > 0.2, f"stone angles genuinely vary around the disc ({spread:.2f} rad)")
    # The lattice, by contrast, is axis-aligned everywhere.
    _pl, _ql, lat = run_lattice()
    check(all(angle_of(t) < 1e-9 for t in lat.mosaic.tesserae),
          "the lattice layout stays square to the page")

    # Neighbouring stones agree in direction — that is a *course*, not noise.
    angles = [angle_of(t) for t in bg.tesserae]
    pts = np.array([[t.cx, t.cy] for t in bg.tesserae])
    agree = 0
    pairs = 0
    for i in range(0, len(bg.tesserae), 3):       # sample, for speed
        d2 = np.sum((pts - pts[i]) ** 2, axis=1)
        d2[i] = np.inf
        j = int(np.argmin(d2))
        diff = abs(angles[i] - angles[j]) % (math.pi / 2.0)
        diff = min(diff, math.pi / 2.0 - diff)
        pairs += 1
        agree += diff < 0.35
    check(agree > 0.7 * pairs,
          f"neighbouring stones line up into courses ({agree}/{pairs})")

    # The distance field is the thing that curves, so check it directly: it is
    # zero on the drawing and rises away from it, never crossing a contour.
    tier = mosaic_flow.contour_tiers(np.asarray(make_plan_scene(), dtype=np.float64), 3)
    dist = mosaic_flow.distance_field(tier)
    check(dist.min() == 0.0, "the field is zero on the drawing")
    check(dist.max() > 1.0, "the field rises away from the drawing")
    edge = np.zeros(tier.shape, dtype=bool)
    edge[:-1, :] |= tier[:-1, :] != tier[1:, :]
    check(float(dist[edge].max()) == 0.0, "every contour pixel sits at distance zero")
    ang = mosaic_flow.course_angles(dist, 3)
    check(ang.shape == dist.shape and np.isfinite(ang).all(), "the angle field is finite")


def group28_mosaic_cartoon():
    print("\n[28] Mosaic — the cartoon")
    _pipe, p, res = run_mosaic()
    g = res.mosaic
    texts = []
    strokes = mosaic_doc.cartoon_strokes(
        g, 0.0, 0.0, 600.0, 900.0, p.grout_frac, "#1b1b1b",
        lambda x, y, s, **kw: texts.append((x, y, s)),
    )
    check(len(strokes) == g.total, "the cartoon outlines every stone")
    check(len(texts) == g.total, "every stone carries its colour number")
    numbers = {int(t[2]) for t in texts}
    check(numbers <= set(range(1, len(g.colors) + 1)),
          "cartoon numbers are all real colour numbers")
    check(all(s.fill is None for s in strokes), "an unhighlighted cartoon is outlines only")
    check(all(s.width > 0 for s in strokes), "cartoon stones are drawn as outlines")
    # Highlighting one colour fills just those stones.
    hi = mosaic_doc.cartoon_strokes(
        g, 0.0, 0.0, 600.0, 900.0, p.grout_frac, "#1b1b1b", lambda *a, **k: None, highlight=0)
    filled = sum(1 for s in hi if s.fill is not None)
    check(filled == g.counts[0], f"highlighting fills exactly colour 1's stones ({filled})")
    # The flow document uses the cartoon; the lattice one uses the lettered grid.
    doc_txt = " ".join(t.text for t in res.scene.layer("mosaic-grid").texts)
    check("A" not in doc_txt.split(), "the flow document has no row letters to count")
    _pl, _ql, lat = run_lattice()
    lat_txt = [t.text for t in lat.scene.layer("mosaic-grid").texts]
    check("A" in lat_txt, "the lattice document still labels its rows")
    # The to-scale sheet keeps one stone == one tessera in both layouts.
    for grid_res, name in ((res, "flow"), (lat, "lattice")):
        sheet = mosaic_doc.build_grid_sheet(p, grid_res.mosaic)
        cell_in = mosaic_doc.SHEET_CELL * sheet.inches_per_px
        check(abs(cell_in - p.tessera_mm / 25.4) < 1e-9,
              f"the {name} sheet prints one cell per tessera")
        check(sheet.width > 0 and sheet.height > 0, f"the {name} sheet has a size")


# --- easel mode ------------------------------------------------------------------------
def group29_easel_picking():
    print("\n[29] Easel — picking a region")
    from studio import easel

    img = make_plan_scene()             # four horizontal bands, 300 x 400
    an = easel.prepare(img)
    check(an.shape[0] > 0 and an.lab.shape[:2] == an.shape, "the reference is prepared once")
    # Clicking inside the second band selects that band — not a speck, not the picture.
    pk = easel.pick_region(an, 0.5, 0.375)
    check(0.18 < pk.area_frac < 0.32, f"a click selects its band ({pk.area_frac * 100:.0f}%)")
    band = np.array(PLAN_BANDS[1], dtype=np.float64)
    de = color.delta_e_2000(color.lab(tuple(int(v) for v in band)), color.lab(color.hex_to_rgb(pk.hex)))
    check(de < 3.0, f"the swatch is that band's colour (ΔE {de:.1f})")
    h, w = an.shape
    check(bool(pk.mask[int(0.375 * h), int(0.5 * w)]), "the region contains the click")
    edge = easel.region_outline(pk.mask)
    check(bool(edge.any()) and not (edge & ~pk.mask).any(), "the outline lies on the region")
    # It never leaks into a neighbouring band of a clearly different colour.
    other = pk.mask[: int(0.22 * h)].mean() + pk.mask[int(0.53 * h):].mean()
    check(other < 0.01, "the region stops at the band's edges")
    # Smaller / larger step monotonically, and a saved tolerance reproduces the pick.
    blob = easel.prepare(make_blob_scene())
    sizes = [easel.pick_region(blob, 0.5, 0.5, bias=b).area_frac for b in (-3, 0, 3)]
    check(sizes[0] <= sizes[1] <= sizes[2], f"region size follows the bias ({sizes})")
    again = easel.pick_region(an, 0.5, 0.375, tolerance=pk.tolerance)
    check(np.array_equal(again.mask, pk.mask), "a saved tolerance redraws the same outline")
    # The auto rule stays out of specks and out of leaks.
    check(easel.natural_index([0.0001, 0.0002, 0.05, 0.06, 0.07, 0.4] + [0.5] * 9) >= 2,
          "auto grows out of a speck")
    idx = easel.natural_index([0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.30] + [0.5] * 7)
    check(idx <= 6, f"auto steps back out of a leak (chose step {idx})")
    # A spot is small and centred on the click.
    sp = easel.pick_spot(an, 0.5, 0.375)
    check(sp.area_frac < 0.01, f"a spot is a small circle ({sp.area_frac * 100:.2f}%)")
    check(sp.tolerance == 0.0, "a spot carries no tolerance")
    crop = easel.crop_around(img, pk.x, pk.y, pk.bbox)
    check(crop.shape == (128, 128, 3), "the history thumbnail is a square crop")
    # Describing a colour in painter's terms.
    # One swatch per area: the same patch re-picked matches; a different one doesn't.
    from types import SimpleNamespace as NS
    saved = NS(pick="region", hex=pk.hex, x=pk.x, y=pk.y, area_frac=pk.area_frac,
               tolerance=pk.tolerance, bias=0)
    again2 = easel.pick_region(an, 0.52, 0.36)                # another click, same band
    check(easel.find_same(an, again2, "region", 0, [saved]) is saved,
          "a second click in the same band is the same area")
    other_band = easel.pick_region(an, 0.5, 0.875)
    check(easel.find_same(an, other_band, "region", 0, [saved]) is None,
          "a different band is a different area")
    check(easel.find_same(an, sp, "spot", 0, [saved]) is None, "a spot never duplicates a region")
    check(easel.find_same(an, easel.pick_region(an, 0.52, 0.36, bias=2), "region", 2, [saved]) is None,
          "a deliberate Size change makes its own swatch")
    saved_spot = NS(pick="spot", hex=sp.hex, x=sp.x, y=sp.y, area_frac=sp.area_frac,
                    tolerance=0.0, bias=0)
    check(easel.find_same(an, easel.pick_spot(an, 0.501, 0.376), "spot", 0, [saved_spot]) is saved_spot,
          "a spot a pixel away is the same spot")
    check(easel.find_same(an, easel.pick_spot(an, 0.2, 0.376), "spot", 0, [saved_spot]) is None,
          "a spot across the band is a different spot")
    # Overlap, not size: a re-click that regrows over a smaller part of the same
    # patch is still that patch.
    half = easel.Pick(hex=pk.hex, mask=pk.mask & (np.arange(pk.mask.shape[1])[None, :] < pk.mask.shape[1] // 3),
                      area_frac=0.0, x=pk.x, y=pk.y, bbox=pk.bbox)
    check(easel.same_area(half, "region", saved, pk.mask), "a smaller region inside the same patch matches")
    # The eyedropper: one exact pixel, not an average — and read at full resolution.
    odd = np.full((300, 400, 3), 120, np.uint8)
    odd[150, 200] = (200, 30, 40)
    oan = easel.prepare(odd)
    ex = easel.pick_pixel(oan, odd, 200.5 / 400, 150.5 / 300)
    check(ex.hex == "#c81e28", f"the eyedropper returns the exact pixel ({ex.hex})")
    check(ex.kind == "pixel" and ex.image_shape == (300, 400), "it is an exact-pixel pick")
    avg = easel.pick_region(oan, 200.5 / 400, 150.5 / 300)
    check(avg.hex != ex.hex, f"where the area average differs ({avg.hex})")
    check(ex.mask.sum() == 1 and ex.mask.shape == oan.shape, "its mask marks one analysis pixel")
    saved_px = NS(pick="pixel", hex=ex.hex, x=ex.x, y=ex.y, bias=0)
    check(easel.find_same(oan, easel.pick_pixel(oan, odd, 201.5 / 400, 150.5 / 300), "pixel", 0,
                          [saved_px]) is saved_px, "an exact pick a pixel away is the same point")
    check(easel.find_same(oan, easel.pick_pixel(oan, odd, 206.5 / 400, 150.5 / 300), "pixel", 0,
                          [saved_px]) is None, "an exact pick a few pixels away is a new point")
    check(easel.find_same(oan, avg, "region", 0, [saved_px]) is None,
          "an area never duplicates an exact pick")
    patch = easel.loupe_patch(odd, 200.5 / 400, 150.5 / 300)
    check(patch.shape == (15, 15, 3) and tuple(patch[7, 7]) == (200, 30, 40),
          "the loupe is centred on the pixel a click takes")
    check(easel.loupe_patch(odd, 0.0, 0.0).shape == (15, 15, 3), "the loupe works at the edge")
    names = ["Swatch 7", "Swatch 6", "Swatch 5", "Swatch 4", "Swatch 3", "Swatch 1"]
    check(easel.next_swatch_name(names) == "Swatch 8",
          "a new swatch never reuses a name after a delete")
    check(easel.next_swatch_name(["cheek", "Swatch 2"]) == "Swatch 3", "renamed swatches are skipped")
    check(easel.next_swatch_name([]) == "Swatch 1", "the first swatch is Swatch 1")
    check("near-grey" in easel.describe("#777777"), "a mid grey is described as near-grey")
    check("warm" in easel.describe("#c0703a"), "an orange is described as warm")
    L, v = easel.value_of("#ffffff")
    check(v == 10 and abs(L - 100) < 0.5, "white is value 10")
    g = easel.grey_of("#c0703a")
    check(abs(color.lstar(color.hex_to_rgb(g)) - color.lstar(color.hex_to_rgb("#c0703a"))) < 1.0,
          "value mode keeps the colour's value")


def group30_easel_amounts():
    print("\n[30] Easel — how much to mix")
    from studio import easel

    check(abs(easel.canvas_cm2(10, 10, "cm") - 100.0) < 1e-9, "canvas area in cm")
    check(abs(easel.canvas_cm2(1, 1, "in") - 2.54 ** 2) < 1e-9, "canvas area in inches")
    check(easel.canvas_cm2(0, 20, "in") == 0.0, "an unset canvas has no area")
    area = easel.canvas_cm2(16, 20, "in")
    small, big = easel.amount_ml(0.02, area), easel.amount_ml(0.2, area)
    check(big > small, "a bigger region needs more paint")
    check(easel.amount_ml(0.2, area * 2) > big, "a bigger canvas needs more paint")
    check(easel.amount_ml(0.0001, area) == easel.MIN_ML, "a tiny region still gets a mixable pea")
    check(easel.amount_ml(0.2, 0.0) == 0.0, "no canvas size means no invented amount")
    piles = [easel.pile_phrase(ml) for ml in (0.3, 1.5, 5, 15, 40)]
    check(piles == ["about a pea", "about a hazelnut", "about a walnut", "about a golf ball",
                    "more than a golf ball — mix it in two batches"], f"piles scale up ({piles})")
    check(easel.tube_text(0.05) == "a tiny touch", "a tiny share reads as a touch")
    check(easel.tube_text(1.0) == "4 cm", f"1 ml is 4 cm of tube ({easel.tube_text(1.0)})")
    ml, text = easel.amount_summary("oil", 0.1, area)
    check(ml > 0 and "ml" in text and "tube" in text and "10% of the canvas" in text,
          f"the oil amount speaks in piles and tube ({text})")
    _ml, unset = easel.amount_summary("oil", 0.1, 0.0)
    check("canvas size" in unset, "an unset canvas asks for its size")
    wml, wtext = easel.amount_summary("watercolor", 0.1, area)
    check(wml > 0 and "brush-load" in wtext, "watercolor amounts reuse the wash volumes")
    check(easel.amount_summary("oilpastel", 0.1, area) == (0.0, ""), "oil pastel has no amount")
    check(easel.paints_per_mix("oil") == 4 and easel.paints_per_mix("watercolor") == 3,
          "oil mixes may use four paints, watercolor three")


def group31_easel_mixing_order():
    print("\n[31] Easel — mixing order")
    from studio import easel
    from studio.models import MixRecipe, MixStep

    by = {p.name: p for p in OIL_STARTER}

    def recipe(*parts):
        return MixRecipe("#888888", "", tuple(MixStep(n, by[n].hex, w) for n, w in parts),
                         "#888888", 1.0)

    # White-dominant: colour goes into white.
    r = recipe(("Yellow Ochre", 0.10), ("Titanium White", 0.85), ("Ultramarine Blue", 0.05))
    order = [s.paint_name for s in easel.ordered_steps(r, OIL_STARTER)]
    check(order[0] == "Titanium White", f"a tint starts from the white ({order})")
    steps = easel.mixing_steps(r, OIL_STARTER, "oil", 4.0)
    check("add colour to it" in steps[0], "the first step says colour goes into white")
    # A strong pigment is never the base, however big its share.
    r = recipe(("Phthalo Blue", 0.45), ("Cadmium Red Light", 0.30),
               ("Titanium White", 0.15), ("Cadmium Yellow Light", 0.10))
    order = [s.paint_name for s in easel.ordered_steps(r, OIL_STARTER)]
    check(order[0] != "Phthalo Blue", f"Phthalo never starts a mix ({order})")
    check(order[-1] == "Phthalo Blue", "the strong pigment goes in last")
    steps = easel.mixing_steps(r, OIL_STARTER, "oil", 3.5)
    check(any("strong pigment" in s and "Phthalo" in s for s in steps),
          "the Phthalo step warns it is strong")
    check(any("touch of Cadmium Yellow" in s for s in steps),
          "a small share is a touch — without being called strong")
    check(not any("Cadmium Yellow" in s and "strong" in s for s in steps),
          "a small share is not mislabelled as strong")
    check(steps[-1].startswith("Check it against the swatch"), "every mix ends with a check")
    # Amounts add up to the total.
    import re
    cms = [float(m) for s in steps for m in re.findall(r"([0-9.]+) cm", s)]
    check(abs(sum(cms) * easel.TUBE_ML_PER_CM - 3.5) < 0.6, f"step amounts sum to the total ({cms})")
    # Without a canvas size the steps fall back to parts.
    steps = easel.mixing_steps(r, OIL_STARTER, "oil", 0.0)
    check(any("parts" in s or "part " in s for s in steps), "no canvas size → parts, not cm")
    # One paint, watercolor and pastel all read correctly.
    single = recipe(("Yellow Ochre", 1.0))
    check(easel.mixing_steps(single, OIL_STARTER, "oil", 2.0)[0].startswith("Use Yellow Ochre straight"),
          "a single paint is used straight from the tube")
    wr = MixRecipe("#aaaaaa", "", (MixStep("French Ultramarine", WATERCOLOR_STARTER[8].hex, 1.0),),
                   "#aaaaaa", 1.0, dilution=0.3)
    wsteps = easel.mixing_steps(wr, WATERCOLOR_STARTER, "watercolor", 5.0)
    check(any("wash" in s for s in wsteps), "watercolor steps speak in washes")
    # A renamed paint is named by its current name.
    renamed = tuple(Paint("My Blue", q.hex, q.is_white, q.opacity, q.staining, q.granulating)
                    if q.name == "Phthalo Blue" else q for q in OIL_STARTER)
    r = recipe(("Phthalo Blue", 0.45), ("Cadmium Red Light", 0.55))
    check(any("My Blue" in s for s in easel.mixing_steps(r, renamed, "oil", 2.0)),
          "steps use the paint's current name")


def group32_library():
    print("\n[32] Easel — the library on disk")
    from studio import easel
    from studio.library import Library, NamedPalette, Swatch, new_swatch_id, palette_or_first

    with tempfile.TemporaryDirectory() as td:
        lib = Library(td)
        pals = lib.palettes()
        check("Oil — split primary 8" in pals and len(pals) == 4, "a fresh library has the starter palettes")
        check(pals["Oil — split primary 8"].paints == OIL_STARTER, "starter palettes are the real kits")
        mine = NamedPalette("Zorn limited", "oil", OIL_STARTER[:4])
        lib.save_palette(mine)
        check(Library(td).palette("Zorn limited").paints == OIL_STARTER[:4], "a saved palette persists")
        check(lib.default_palette_name() == "", "no palette is 'yours' until you choose one")
        lib.set_default_palette("Zorn limited")
        check(Library(td).default_palette_name() == "Zorn limited", "your palette is remembered")
        lib.save_palette(NamedPalette("Another", "oil", OIL_STARTER[:2]))
        check(lib.default_palette_name() == "Zorn limited", "saving another palette keeps yours")

        ref = os.path.join(td, "photo.png")
        from PIL import Image
        Image.fromarray(make_plan_scene()).save(ref)
        proj = lib.create_project("Lemons", "Zorn limited", ref, 40, 50, "cm")
        check(os.path.isfile(lib.reference_path(proj)), "the reference is copied into the painting")
        os.remove(ref)
        check(os.path.isfile(lib.reference_path(Library(td).load_project(proj.id))),
              "deleting the original photo doesn't break the painting")

        an = easel.prepare(make_plan_scene())
        pk = easel.pick_region(an, 0.5, 0.375)
        from studio.pipeline.solver import solve_targets
        r = solve_targets([Target(hex=pk.hex)], OIL_STARTER[:4], "oil", 4)[0]
        sw = Swatch(id=new_swatch_id(), hex=pk.hex, x=pk.x, y=pk.y, area_frac=pk.area_frac,
                    recipe=r, palette="Zorn limited", medium="oil", tolerance=pk.tolerance,
                    name="sky", notes="needs more ochre", tags=("background",))
        lib.add_swatch(proj, sw, easel.crop_around(make_plan_scene(), pk.x, pk.y, pk.bbox))
        back = Library(td).load_project(proj.id)
        got = back.swatch(sw.id)
        check(got is not None and got.recipe == r, "a swatch's recipe survives a restart exactly")
        check((got.name, got.notes, got.tags) == ("sky", "needs more ochre", ("background",)),
              "name, notes and tags survive a restart")
        check((back.canvas_w, back.canvas_h, back.units) == (40.0, 50.0, "cm"), "the canvas survives")
        check(lib.crop_path(back, got) is not None, "the crop is saved beside it")
        check(not [f for f in os.listdir(back.path) if ".tmp-" in f], "no temp files are left behind")

        # A second painting, and searching across them.
        other = lib.create_project("Portrait", "Oil — split primary 8")
        lib.add_swatch(other, Swatch(id=new_swatch_id(), hex="#c9a38a", x=0.5, y=0.5, area_frac=0.1,
                                     recipe=r, palette="Oil — split primary 8", medium="oil",
                                     name="cheek light"))
        check([s.name for _p, s in lib.search("cheek")] == ["cheek light"], "search finds by name")
        check(len(lib.search("ochre")) == 1, "search finds by notes")
        check(len(lib.search("background")) == 1, "search finds by tag")
        check(len(lib.search("lemons")) == 1, "search finds by painting")
        paint = r.steps[0].paint_name
        check(len(lib.search(paint)) == 2, "search finds by paint across paintings")
        check(len(lib.search("", proj.id)) == 1, "a painting's history holds only its own swatches")
        check([p.name for p in lib.projects()][0] == "Portrait", "the latest painting is listed first")

        # Palettes may go away; swatches keep their recipe, lookups fall back.
        lib.delete_palette("Zorn limited")
        check(lib.default_palette_name() == "", "deleting your palette clears the choice cleanly")
        check(Library(td).load_project(proj.id).swatches[0].recipe == r,
              "deleting a palette leaves old recipes alone")
        check(palette_or_first(lib, "Zorn limited", "oil").medium == "oil",
              "a deleted palette falls back to one of the same medium")

        # Deleting cleans up, and only ever touches real paintings.
        lib.delete_swatch(proj, sw.id)
        check(not os.listdir(os.path.join(proj.path, "crops")), "deleting a swatch removes its crop")
        stray = os.path.join(lib.projects_dir, "not-a-painting")
        os.makedirs(stray)
        lib.delete_project("not-a-painting")
        check(os.path.isdir(stray), "a folder without project.json is never deleted")
        with open(os.path.join(stray, "project.json"), "w") as fh:
            fh.write("{ broken")
        check(len(lib.projects()) == 2, "a damaged painting never hides the others")
        lib.delete_project(other.id)
        check(not os.path.exists(other.path), "deleting a painting removes its folder")


def group33_solver_four_paints():
    print("\n[33] Solver — four-paint mixes and the symmetric grid")
    from studio.pipeline import solver

    firsts = {a for a, _b, _c in solver._TRIPLE_GRID}
    check(max(firsts) >= 0.8, "no slot of a three-paint mix is capped at 60%")
    check(all(abs(sum(w) - 1.0) < 1e-9 for w in solver._TRIPLE_GRID + solver._QUAD_GRID),
          "every grid point sums to one")
    # Light tints: white can finally dominate a three-paint oil mix.
    r = solve_targets([Target(hex="#d7d0c5")], OIL_STARTER, "oil", 3)[0]
    white = sum(s.weight for s in r.steps if s.paint_name == "Titanium White")
    check(white >= 0.7, f"a light tint is mostly white ({white:.2f})")
    # Neutrals: a fourth paint closes the gap.
    for hx in ("#998f81", "#b3a89a"):
        r3 = solve_targets([Target(hex=hx)], OIL_STARTER, "oil", 3)[0]
        r4 = solve_targets([Target(hex=hx)], OIL_STARTER, "oil", 4)[0]
        check(r4.delta_e < r3.delta_e - 2.0 and r4.delta_e < 3.5,
              f"four paints reach the grey {hx} (ΔE {r3.delta_e:.1f} → {r4.delta_e:.1f})")
        check(len(r4.steps) <= 4, "a four-paint recipe uses at most four paints")
    again = solve_targets([Target(hex="#998f81")], OIL_STARTER, "oil", 4)[0]
    check(again == solve_targets([Target(hex="#998f81")], OIL_STARTER, "oil", 4)[0],
          "four-paint solving is deterministic")


def group34_easel_window():
    print("\n[34] Easel — the window")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication, QMessageBox
    from PySide6.QtCore import QSettings

    app = QApplication.instance() or QApplication([])
    app.setOrganizationName("StudioSmokeTest")
    app.setApplicationName("StudioSmokeTest")
    QSettings().clear()
    from studio import easel
    from studio.library import Library
    from studio.ui.easel_window import EaselWindow

    with tempfile.TemporaryDirectory() as td:
        lib = Library(td)
        empty = EaselWindow(library=lib)
        check(empty.project is None and empty.canvas.mode == "empty",
              "an empty library asks for a first painting")
        empty.close()

        ref = os.path.join(td, "ref.png")
        from PIL import Image
        Image.fromarray(make_plan_scene()).save(ref)
        proj = lib.create_project("Bands", "Oil — split primary 8", ref, 16, 20, "in")
        win = EaselWindow(library=lib)
        win.resize(1200, 800)
        win.show()
        app.processEvents()
        check(win.project is not None and win.project.id == proj.id, "the painting opens")
        check(win.canvas.mode == "photo" and win.analysis is not None, "the reference is shown")

        # Click -> a big swatch, saved as you go.
        win._on_picked(0.5, 0.375, False)
        app.processEvents()
        check(win.canvas.mode == "swatch", "a click shows the colour big")
        saved = Library(td).load_project(proj.id)
        check(len(saved.swatches) == 1, "the swatch is saved to disk immediately")
        sw = saved.swatches[0]
        check(sw.recipe is not None and sw.palette == "Oil — split primary 8",
              "it carries the recipe and the palette it was mixed with")
        check(lib.crop_path(saved, sw) is not None, "and a crop of where it came from")
        card = win.canvas._card
        check(card.ratio_text and card.steps and "ml" in card.amount_text,
              "the card shows the ratio, the steps and the amount")
        check(win.history.count() == 1, "the history lists it")

        # Click again -> back to the photo, region still outlined.
        win._back_to_photo()
        check(win.canvas.mode == "photo" and win.canvas._overlay is not None,
              "going back shows the photo with the region outlined")

        # The same area again opens the existing swatch — no copy.
        first_id = win.current
        win._on_picked(0.51, 0.37, False)
        check(len(Library(td).load_project(proj.id).swatches) == 1,
              "clicking the same area again doesn't add a copy")
        check(win.canvas.mode == "swatch" and win.current == first_id,
              "it shows the swatch that's already there")
        win._back_to_photo()

        # Clearing the selection: Esc from the photo, ⌘D, or the margin.
        win._escape()
        check(not win.canvas.has_region and win.current is None,
              "Esc on the photo clears the selection — no dimming")
        check(win.history.currentRow() == -1 and win.name_edit.text() == "",
              "and nothing is selected in the history")
        win._on_history_clicked(win.history.item(0))
        win._escape()
        check(win.canvas.mode == "photo" and win.canvas.has_region,
              "Esc from a swatch steps back to the photo first")
        win.act_clear.trigger()
        check(not win.canvas.has_region, "Clear Selection (⌘D) clears it")
        win._on_history_clicked(win.history.item(0))
        win._back_to_photo()
        win.canvas.deselected.emit()
        check(not win.canvas.has_region, "clicking the margin around the photo clears it")
        # Option-click from Area takes the exact pixel.
        win._on_picked(0.3, 0.375, True)
        exact = win.project.swatches[0]
        iy, ix = easel.pixel_at(win.display_rgb, 0.3, 0.375)
        check(exact.pick == "pixel" and exact.hex == color.rgb_to_hex(tuple(int(c) for c in win.display_rgb[iy, ix])),
              "Option-click takes the exact pixel, not an average")
        check("no area" in win.canvas._card.amount_text, "an exact point says it can't size an amount")
        win._back_to_photo()
        check(win.canvas._marker is not None and win.canvas._overlay is None,
              "an exact pick is ringed on the photo, nothing dimmed")
        n_before = len(win.project.swatches)
        win._on_picked(0.3005, 0.375, True)
        check(len(win.project.swatches) == n_before, "the same exact point isn't saved twice")
        win._back_to_photo()
        # The picker: E toggles, Size only applies to areas, the loupe follows the eyedropper.
        win._toggle_eyedropper()
        check(win.pick_mode.currentData() == "pixel", "E switches to the eyedropper")
        check(not win.region_bias.isEnabled(), "Size is disabled for the eyedropper")
        check(win.canvas._loupe_mode, "the loupe is on in eyedropper mode")
        win._on_picked(0.7, 0.375, False)
        check(win.project.swatches[0].pick == "pixel", "a plain click in eyedropper mode is exact")
        win._back_to_photo()
        check(win._kind_for_click(True) == "region", "Option-click from the eyedropper takes the area")
        win._on_picked(0.52, 0.375, True)
        check(win.current is not None and win.project.swatch(win.current).pick == "region",
              "…which is the band already saved, so it shows that swatch")
        win._back_to_photo()
        from PySide6.QtCore import QPointF as _P
        win.canvas._hover = _P(win.canvas._rect.center())
        win.canvas.grab()                     # paints the loupe without error
        check(win.canvas._hover is not None, "the loupe paints over the photo")
        win.canvas._hover = None
        win._toggle_eyedropper()
        check(win.pick_mode.currentData() == "region" and win.region_bias.isEnabled()
              and not win.canvas._loupe_mode, "E switches back to Area")
        # A small spot is still there as a third pick.
        win.pick_mode.setCurrentIndex(win.pick_mode.findData("spot"))
        win._on_picked(0.2, 0.875, False)
        check(win.project.swatches[0].pick == "spot", "the small spot is still available")
        win.pick_mode.setCurrentIndex(win.pick_mode.findData("region"))
        win._back_to_photo()

        # Value mode.
        win.act_value.setChecked(True)
        check(win.canvas._value and win.canvas._photo_grey is not None, "V shows value only")
        win.act_value.setChecked(False)

        # Naming, tags, notes persist.
        item = next(win.history.item(i) for i in range(win.history.count())
                    if win.project.swatch(win.history.item(i).data(0x0100)[1]).pick == "region")
        win._on_history_clicked(item)
        win.name_edit.setText("sky band")
        win.tags_edit.setText("background, cool")
        win.notes_edit.setPlainText("glaze later")
        win._save_details()
        named = Library(td).load_project(proj.id).swatch(win.current)
        check(named.name == "sky band" and named.tags == ("background", "cool")
              and named.notes == "glaze later", "name, tags and notes are saved")
        # Notes typed without leaving the field survive switching swatch.
        win.notes_edit.setPlainText("glaze later, thinly")
        win._on_history_clicked(win.history.item(0))
        check(Library(td).load_project(proj.id).swatch(named.id).notes == "glaze later, thinly",
              "a half-typed note is kept when you move to another swatch")

        # Search.
        win.search.setText("sky")
        check(win.history.count() == 1, "search narrows the history")
        win.search.setText("")

        # Changing the palette leaves old recipes alone until re-mixed.
        win.palette_combo.setCurrentText("Acrylic — basic 6")
        check(Library(td).load_project(proj.id).palette == "Acrylic — basic 6",
              "the painting remembers its palette")
        before = win.project.swatch(named.id).recipe
        win._on_history_clicked(next(win.history.item(i) for i in range(win.history.count())
                                     if win.history.item(i).data(0x0100)[1] == named.id))
        check(win.project.swatch(named.id).recipe == before, "old swatches keep their recipe")
        win._remix()
        remixed = Library(td).load_project(proj.id).swatch(named.id)
        check(remixed.palette == "Acrylic — basic 6", "Re-mix uses the current palette")

        # Editing the palette saves it to the library.
        paints = list(win.palette_panel.palette())
        paints[0] = Paint("Studio White", paints[0].hex, paints[0].is_white, paints[0].opacity)
        win._on_palette_edited(tuple(paints))
        check(Library(td).palette("Acrylic — basic 6").paints[0].name == "Studio White",
              "palette edits are saved")

        # Deleting (confirmation answered yes).
        original = QMessageBox.question
        QMessageBox.question = staticmethod(lambda *a, **k: QMessageBox.Yes)
        try:
            n = len(win.project.swatches)
            win._delete_current()
        finally:
            QMessageBox.question = original
        check(len(Library(td).load_project(proj.id).swatches) == n - 1, "delete removes the swatch")
        check(win.canvas.mode == "photo", "and returns to the photo")

        # All paintings.
        other = lib.create_project("Second", "Oil — split primary 8", None)
        win._reload_projects(select=proj.id)
        win.scope.setCurrentIndex(1)
        check(win.history.count() == len(win.project.swatches), "all-paintings scope lists everything")
        win.close()

    # Your palette carries into the main window: first launch and Reset.
    from studio.library import NamedPalette
    envlib = Library(os.environ["PAINTING_STUDIO_LIBRARY"])
    envlib.save_palette(NamedPalette("Test mine", "oil", OIL_STARTER[:3]))
    envlib.set_default_palette("Test mine")
    QSettings().clear()
    from studio.ui.main_window import MainWindow
    mw = MainWindow()
    check(mw.params.palette == OIL_STARTER[:3], "a first launch starts from your palette")
    check(mw.palette_panel.palette() == OIL_STARTER[:3], "and the palette panel shows it")
    mw.params = mw.params.copy(palette=OIL_STARTER)
    mw._reset()
    check(mw.params.palette == OIL_STARTER[:3], "Reset falls back to your palette, not a starter kit")
    ew = EaselWindow(library=envlib)
    check(ew.palette.name == "Test mine", "the easel starts from your palette")
    ew.close()
    mw._thread.quit()
    mw._thread.wait(3000)
    envlib.set_default_palette("")       # leave the shared test library neutral
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
    group18_plan_segmentation()
    group19_plan_schedule()
    group20_plan_document()
    group21_plan_cache()
    group22_mosaic_grid()
    group23_mosaic_tesserae()
    group24_mosaic_document()
    group25_mosaic_tiling()
    group26_mosaic_cache()
    group27_mosaic_andamento()
    group28_mosaic_cartoon()
    group29_easel_picking()
    group30_easel_amounts()
    group31_easel_mixing_order()
    group32_library()
    group33_solver_four_paints()
    group34_easel_window()

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
