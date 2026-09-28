"""The easel's library on disk: named palettes, and one folder per painting.

Layout under the library root (``~/Documents/Painting Studio`` by default)::

    palettes.json                  every named palette
    projects/<id>/project.json     the painting: name, canvas, palette, swatches
    projects/<id>/reference.jpg    a *copy* of the reference photo
    projects/<id>/crops/<id>.png   where on the photo each swatch came from

Plain JSON and PNG on purpose — the files are readable, back up with anything,
and could be opened by a different front end later without a migration.

Three rules keep it trustworthy over months of use:

* **The reference is copied in.** Moving or deleting the original photo never
  breaks an old painting.
* **A swatch keeps the recipe it was mixed with.** Editing or deleting a
  palette later doesn't rewrite history; "re-mix" is an explicit action.
* **Writes are atomic** (temp file + rename), so a crash mid-save can't leave a
  half-written ``project.json`` and lose a painting's history.
* **Each window writes only its own part.** A project is shared: the Easel owns
  its swatches, palette and canvas; the document window owns ``document`` (its
  board / plan / mosaic settings). They can run as separate programs, so every
  save re-reads the other part from disk rather than writing back a stale copy
  — otherwise saving in one would silently erase work done in the other.

Qt-free, so the smoke suite can exercise it against a temp directory.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .models import MixRecipe, MixStep, Paint, paint_from_dict, paint_to_dict
from .palettes import STARTERS
from .params import UNIT_IN

SCHEMA = 1
DEFAULT_ROOT = os.path.join(os.path.expanduser("~"), "Documents", "Painting Studio")

# iPhone photos; converted to JPEG on import (see ``Library.set_reference``).
HEIC_EXTS = {".heic", ".heif"}

# The palettes a fresh library starts with — the same starter kits the board uses.
STARTER_NAMES = {
    "oil": "Oil — split primary 8",
    "watercolor": "Watercolor — 12",
    "acrylic": "Acrylic — basic 6",
    "oilpastel": "Oil pastel — 24",
}


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _atomic_write(path: str, text: str) -> None:
    tmp = f"{path}.tmp-{uuid.uuid4().hex[:6]}"
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(text)
    os.replace(tmp, path)


def _slug(name: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    return (s or "painting")[:40]


# --- models ---------------------------------------------------------------------
@dataclass
class NamedPalette:
    name: str
    medium: str
    paints: Tuple[Paint, ...]


@dataclass
class Swatch:
    """One colour picked off the reference, and how to mix it."""

    id: str
    hex: str                         # the region's average colour — what you match
    x: float                         # where it was clicked, normalized 0..1
    y: float
    area_frac: float                 # the region's share of the picture
    recipe: Optional[MixRecipe]
    palette: str                     # the palette it was mixed with
    medium: str
    pick: str = "region"             # "region" | "spot" — enough to redraw the outline
    tolerance: float = 10.0
    bias: int = 0                    # the Size setting it was picked at
    name: str = ""
    notes: str = ""
    tags: Tuple[str, ...] = ()
    created: str = ""
    crop: str = ""                   # file name inside the project's crops/ folder

    def label(self, index: int = 0) -> str:
        return self.name or f"Swatch {index}"


@dataclass
class Project:
    """One painting."""

    id: str                          # folder name
    name: str
    palette: str
    canvas_w: float = 0.0
    canvas_h: float = 0.0
    units: str = UNIT_IN
    reference: str = ""              # file name inside the project folder
    tolerance: float = 10.0
    swatches: List[Swatch] = field(default_factory=list)
    created: str = ""
    updated: str = ""
    # The document window's saved state: {"params": …, "page": int, "dpi": int}.
    # None until the project has been saved from there.
    document: Optional[dict] = None
    path: str = ""                   # absolute folder; not serialized

    def swatch(self, sid: str) -> Optional[Swatch]:
        return next((s for s in self.swatches if s.id == sid), None)


# --- (de)serialization ---------------------------------------------------------
def _recipe_to_dict(r: Optional[MixRecipe]) -> Optional[dict]:
    if r is None:
        return None
    return {
        "target_hex": r.target_hex,
        "target_label": r.target_label,
        "steps": [{"paint_name": s.paint_name, "paint_hex": s.paint_hex, "weight": s.weight}
                  for s in r.steps],
        "mixed_hex": r.mixed_hex,
        "delta_e": r.delta_e,
        "dilution": r.dilution,
        "layering": r.layering,
        "note": r.note,
        "achievable": r.achievable,
    }


def _recipe_from_dict(d: Optional[dict]) -> Optional[MixRecipe]:
    if not d:
        return None
    return MixRecipe(
        target_hex=d.get("target_hex", ""),
        target_label=d.get("target_label", ""),
        steps=tuple(
            MixStep(s.get("paint_name", "?"), s.get("paint_hex", "#000000"), float(s.get("weight", 0)))
            for s in d.get("steps", [])
        ),
        mixed_hex=d.get("mixed_hex", "#000000"),
        delta_e=float(d.get("delta_e", 0.0)),
        dilution=float(d.get("dilution", 1.0)),
        layering=bool(d.get("layering", False)),
        note=d.get("note", ""),
        achievable=bool(d.get("achievable", True)),
    )


def _swatch_to_dict(s: Swatch) -> dict:
    return {
        "id": s.id, "hex": s.hex, "x": s.x, "y": s.y, "area_frac": s.area_frac,
        "recipe": _recipe_to_dict(s.recipe), "palette": s.palette, "medium": s.medium,
        "pick": s.pick, "tolerance": s.tolerance, "bias": s.bias, "name": s.name, "notes": s.notes,
        "tags": list(s.tags), "created": s.created, "crop": s.crop,
    }


def _swatch_from_dict(d: dict) -> Swatch:
    return Swatch(
        id=str(d.get("id") or uuid.uuid4().hex[:10]),
        hex=d.get("hex", "#000000"),
        x=float(d.get("x", 0.5)),
        y=float(d.get("y", 0.5)),
        area_frac=float(d.get("area_frac", 0.0)),
        recipe=_recipe_from_dict(d.get("recipe")),
        palette=d.get("palette", ""),
        medium=d.get("medium", "oil"),
        pick=d.get("pick", "region"),
        tolerance=float(d.get("tolerance", 10.0)),
        bias=int(d.get("bias", 0)),
        name=d.get("name", ""),
        notes=d.get("notes", ""),
        tags=tuple(d.get("tags", ())),
        created=d.get("created", ""),
        crop=d.get("crop", ""),
    )


def _project_to_dict(p: Project) -> dict:
    return {
        "schema": SCHEMA, "id": p.id, "name": p.name, "palette": p.palette,
        "canvas_w": p.canvas_w, "canvas_h": p.canvas_h, "units": p.units,
        "reference": p.reference, "tolerance": p.tolerance,
        "created": p.created, "updated": p.updated,
        "swatches": [_swatch_to_dict(s) for s in p.swatches],
        "document": p.document,
    }


def _project_from_dict(d: dict, path: str) -> Project:
    return Project(
        id=d.get("id") or os.path.basename(path),
        name=d.get("name", "Untitled"),
        palette=d.get("palette", ""),
        canvas_w=float(d.get("canvas_w", 0.0)),
        canvas_h=float(d.get("canvas_h", 0.0)),
        units=d.get("units", UNIT_IN),
        reference=d.get("reference", ""),
        tolerance=float(d.get("tolerance", 10.0)),
        swatches=[_swatch_from_dict(s) for s in d.get("swatches", [])],
        created=d.get("created", ""),
        updated=d.get("updated", ""),
        document=d.get("document") if isinstance(d.get("document"), dict) else None,
        path=path,
    )


# --- the library ----------------------------------------------------------------
class Library:
    """Named palettes and painting projects under one folder."""

    def __init__(self, root: str = DEFAULT_ROOT):
        self.root = root
        self.projects_dir = os.path.join(root, "projects")
        os.makedirs(self.projects_dir, exist_ok=True)
        self._palettes_path = os.path.join(root, "palettes.json")
        if not os.path.exists(self._palettes_path):
            self._write_palettes({
                STARTER_NAMES[m]: NamedPalette(STARTER_NAMES[m], m, paints)
                for m, paints in STARTERS.items()
            })

    # -- palettes -----------------------------------------------------------------
    def _read_palettes_raw(self) -> dict:
        try:
            with open(self._palettes_path, encoding="utf-8") as fh:
                data = json.load(fh)
            return data if isinstance(data, dict) else {}
        except (OSError, ValueError):
            return {}

    def palettes(self) -> Dict[str, NamedPalette]:
        data = self._read_palettes_raw()
        out: Dict[str, NamedPalette] = {}
        for name, entry in (data.get("palettes") or {}).items():
            out[name] = NamedPalette(
                name=name,
                medium=entry.get("medium", "oil"),
                paints=tuple(paint_from_dict(p) for p in entry.get("paints", [])),
            )
        return out

    def palette(self, name: str) -> Optional[NamedPalette]:
        return self.palettes().get(name)

    def _write_palettes(self, pals: Dict[str, NamedPalette], default: Optional[str] = None) -> None:
        if default is None:
            default = self._read_palettes_raw().get("default", "")
        data = {
            "schema": SCHEMA,
            "default": default,
            "palettes": {
                n: {"medium": p.medium, "paints": [paint_to_dict(x) for x in p.paints]}
                for n, p in pals.items()
            },
        }
        _atomic_write(self._palettes_path, json.dumps(data, indent=2))

    def save_palette(self, pal: NamedPalette) -> None:
        pals = self.palettes()
        pals[pal.name] = pal
        self._write_palettes(pals)

    def delete_palette(self, name: str) -> None:
        pals = self.palettes()
        pals.pop(name, None)
        self._write_palettes(pals)

    def default_palette_name(self) -> str:
        """The artist's own palette — what new paintings and a reset start from.

        Empty when none has been chosen (or the chosen one was since deleted),
        in which case callers keep the starter palettes.
        """
        data = self._read_palettes_raw()
        name = data.get("default", "")
        return name if name in (data.get("palettes") or {}) else ""

    def set_default_palette(self, name: str) -> None:
        self._write_palettes(self.palettes(), default=name)

    # -- projects -------------------------------------------------------------------
    def projects(self) -> List[Project]:
        """Every painting, most recently worked on first."""
        out: List[Project] = []
        for entry in sorted(os.listdir(self.projects_dir)):
            try:
                out.append(self.load_project(entry))
            except (OSError, ValueError, KeyError):
                continue  # a damaged folder must never hide the others
        out.sort(key=lambda p: (p.updated or p.created, p.name), reverse=True)
        return out

    def load_project(self, pid: str) -> Project:
        path = os.path.join(self.projects_dir, pid)
        with open(os.path.join(path, "project.json"), encoding="utf-8") as fh:
            return _project_from_dict(json.load(fh), path)

    def create_project(
        self,
        name: str,
        palette: str,
        reference_src: Optional[str] = None,
        canvas_w: float = 0.0,
        canvas_h: float = 0.0,
        units: str = UNIT_IN,
    ) -> Project:
        pid = f"{_slug(name)}-{uuid.uuid4().hex[:6]}"
        path = os.path.join(self.projects_dir, pid)
        os.makedirs(os.path.join(path, "crops"), exist_ok=True)
        now = _now()
        proj = Project(id=pid, name=name.strip() or "Untitled", palette=palette,
                       canvas_w=canvas_w, canvas_h=canvas_h, units=units,
                       created=now, updated=now, path=path)
        if reference_src:
            self.set_reference(proj, reference_src, save=False)
        self.save_project(proj)
        return proj

    def save_project(self, proj: Project, keep_document: bool = True) -> None:
        """Write a project. ``document`` is taken from disk unless this save owns it.

        The Easel saves through here on every swatch; if it wrote back the
        ``document`` it loaded hours ago it would undo the document window's
        saves. For the same reason a reference file that has vanished from under
        a stale in-memory copy is replaced by whatever the file on disk names.
        """
        path = os.path.join(proj.path, "project.json")
        disk: dict = {}
        if os.path.isfile(path):
            try:
                with open(path, encoding="utf-8") as fh:
                    disk = json.load(fh) or {}
            except (OSError, ValueError):
                disk = {}
        if keep_document and disk:
            proj.document = disk.get("document") if isinstance(disk.get("document"), dict) else None
        on_disk_ref = disk.get("reference", "")
        if (proj.reference and on_disk_ref and on_disk_ref != proj.reference
                and not os.path.isfile(os.path.join(proj.path, proj.reference))
                and os.path.isfile(os.path.join(proj.path, on_disk_ref))):
            proj.reference = on_disk_ref
        proj.updated = _now()
        _atomic_write(path, json.dumps(_project_to_dict(proj), indent=2))

    def save_document(self, pid: str, document: dict, reference_src: Optional[str] = None) -> Project:
        """Save the document window's part of a project, leaving the Easel's alone.

        Loads the project fresh from disk first, so swatches the Easel made since
        the document window opened it are kept. ``reference_src`` replaces the
        reference photo (copied in) when the document window opened a new one.
        """
        proj = self.load_project(pid)
        if reference_src:
            self.set_reference(proj, reference_src, save=False)
        proj.document = document
        self.save_project(proj, keep_document=False)
        return proj

    def duplicate_project(self, pid: str, name: str) -> Project:
        """A full copy of a project under a new name — what Save As does."""
        src = os.path.join(self.projects_dir, pid)
        new_id = f"{_slug(name)}-{uuid.uuid4().hex[:6]}"
        dst = os.path.join(self.projects_dir, new_id)
        shutil.copytree(src, dst)
        proj = self.load_project(new_id)
        proj.id, proj.name, proj.path = new_id, name.strip() or "Untitled", dst
        proj.created = _now()
        self.save_project(proj)
        return proj

    def delete_project(self, pid: str) -> None:
        path = os.path.join(self.projects_dir, pid)
        # Only ever remove a folder that is really one of ours.
        if os.path.isfile(os.path.join(path, "project.json")):
            shutil.rmtree(path)

    def set_reference(self, proj: Project, src: str, save: bool = True) -> str:
        """Copy a photo into the project as its reference (replacing any old one).

        iPhone photos arrive as HEIC, which Pillow can't read without an extra
        package; macOS's own ``sips`` converts them to JPEG on the way in.
        """
        ext = os.path.splitext(src)[1].lower() or ".jpg"
        if ext in HEIC_EXTS:
            name = "reference.jpg"
            dst = os.path.join(proj.path, name)
            try:
                subprocess.run(["sips", "-s", "format", "jpeg", src, "--out", dst],
                               check=True, capture_output=True, timeout=60)
            except (OSError, subprocess.SubprocessError) as exc:
                raise ValueError(
                    "Couldn't convert that HEIC photo. Export it as JPEG from Photos "
                    "(File → Export) and choose that instead.") from exc
        else:
            name = f"reference{ext}"
            dst = os.path.join(proj.path, name)
            if os.path.abspath(src) != os.path.abspath(dst):
                shutil.copy2(src, dst)
        if proj.reference and proj.reference != name:
            old = os.path.join(proj.path, proj.reference)
            if os.path.isfile(old) and os.path.basename(old).startswith("reference"):
                os.remove(old)
        proj.reference = name
        if save:
            self.save_project(proj)
        return dst

    def reference_path(self, proj: Project) -> Optional[str]:
        if not proj.reference:
            return None
        path = os.path.join(proj.path, proj.reference)
        return path if os.path.isfile(path) else None

    # -- swatches -----------------------------------------------------------------------
    def add_swatch(self, proj: Project, swatch: Swatch, crop_rgb: Optional[np.ndarray] = None) -> Swatch:
        if not swatch.created:
            swatch.created = _now()
        if crop_rgb is not None:
            from PIL import Image

            os.makedirs(os.path.join(proj.path, "crops"), exist_ok=True)
            swatch.crop = f"{swatch.id}.png"
            Image.fromarray(np.ascontiguousarray(crop_rgb)).save(
                os.path.join(proj.path, "crops", swatch.crop))
        proj.swatches.insert(0, swatch)          # newest first
        self.save_project(proj)
        return swatch

    def delete_swatch(self, proj: Project, sid: str) -> None:
        sw = proj.swatch(sid)
        if sw is None:
            return
        if sw.crop:
            crop = os.path.join(proj.path, "crops", sw.crop)
            if os.path.isfile(crop):
                os.remove(crop)
        proj.swatches = [s for s in proj.swatches if s.id != sid]
        self.save_project(proj)

    def crop_path(self, proj: Project, swatch: Swatch) -> Optional[str]:
        if not swatch.crop:
            return None
        path = os.path.join(proj.path, "crops", swatch.crop)
        return path if os.path.isfile(path) else None

    def search(self, query: str, project_id: Optional[str] = None) -> List[Tuple[Project, Swatch]]:
        """Swatches matching ``query`` in name, notes, tags, paints, painting or hex.

        An empty query lists everything in scope, newest first.
        """
        q = query.strip().lower()
        projects = [self.load_project(project_id)] if project_id else self.projects()
        out: List[Tuple[Project, Swatch]] = []
        for proj in projects:
            for sw in proj.swatches:
                if not q or q in _haystack(proj, sw):
                    out.append((proj, sw))
        out.sort(key=lambda ps: ps[1].created, reverse=True)
        return out


def new_swatch_id() -> str:
    return uuid.uuid4().hex[:10]


def _haystack(proj: Project, sw: Swatch) -> str:
    paints = " ".join(s.paint_name for s in (sw.recipe.steps if sw.recipe else ()))
    return " ".join((sw.name, sw.notes, " ".join(sw.tags), paints, proj.name, sw.hex, sw.palette)).lower()


def palette_or_first(lib: Library, name: str, medium: str = "oil") -> NamedPalette:
    """The named palette, or a sensible fallback if it was deleted since."""
    pals = lib.palettes()
    if name in pals:
        return pals[name]
    same = [p for p in pals.values() if p.medium == medium]
    if same:
        return same[0]
    if pals:
        return next(iter(pals.values()))
    starter = STARTER_NAMES.get(medium, STARTER_NAMES["oil"])
    return NamedPalette(starter, medium, STARTERS.get(medium, STARTERS["oil"]))
