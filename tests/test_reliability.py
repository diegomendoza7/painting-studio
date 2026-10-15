"""Regression checks for source transitions, exports, and shutdown."""
from __future__ import annotations
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ["XDG_CONFIG_HOME"] = tempfile.mkdtemp(prefix="reliability-config-")
os.environ["BARGUE_DATA_DIR"] = tempfile.mkdtemp(prefix="reliability-bargue-")
os.environ["PAINTING_STUDIO_LIBRARY"] = tempfile.mkdtemp(prefix="reliability-studio-")
import numpy as np
from PIL import Image
from PySide6.QtWidgets import QApplication, QFileDialog, QMessageBox
from PySide6.QtCore import QSettings
from studio.ui.main_window import MainWindow

def wait_for(condition, limit=10):
    end = time.monotonic() + limit
    while time.monotonic() < end:
        QApplication.processEvents()
        if condition():
            return
        time.sleep(.005)
    raise AssertionError("Timed out waiting for application state")

class WindowReliability(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])
        cls.app.setOrganizationName("ReliabilityTests")
        cls.app.setApplicationName("Studio")
        cls.app.setQuitOnLastWindowClosed(False)

    def setUp(self):
        QSettings().clear()
        self.tmp = tempfile.TemporaryDirectory()
        self.photo = Path(self.tmp.name) / "first.png"
        rgb = np.random.default_rng(10).integers(40, 210, (64, 128, 3), dtype=np.uint8)
        Image.fromarray(rgb).save(self.photo)
        self.other = Path(self.tmp.name) / "second.png"
        Image.fromarray(rgb[:, ::-1]).save(self.other)
        self.window = MainWindow()
        self.window.params = self.window.params.copy(palette=self.window.params.palette[:3])
        self.window._load_path(str(self.photo))
        self.idle()

    def idle(self):
        w = self.window
        wait_for(lambda: w._result is not None and not w._busy and not w._pending
                 and not w._loading_file and not w._timer.isActive())

    def tearDown(self):
        w = self.window
        w.close()
        wait_for(lambda: w._shutdown_done and not w._thread.isRunning())
        QApplication.processEvents()
        self.tmp.cleanup()

    def test_failed_and_superseded_loads_preserve_committed_source(self):
        w = self.window
        missing = str(Path(self.tmp.name) / "missing.png")
        with patch.object(QMessageBox, "warning") as warning:
            w._load_path(str(self.other))
            w._load_path(missing)
            self.assertEqual(w._path, str(self.photo))
            self.idle()
            self.assertEqual(w._path, str(self.photo))
            self.assertEqual(w._worker.pipeline._raw.source_path, str(self.photo))
            self.assertEqual(warning.call_count, 1)
            self.assertTrue(w._export_ready())

    def test_pdf_selection_during_load_commits_requested_page_and_dpi(self):
        import fitz
        w = self.window
        if hasattr(w.params, "extract_enabled"):
            w.params = w.params.copy(extract_enabled=False)
        pdf = Path(self.tmp.name) / "pages.pdf"
        with fitz.open() as doc:
            for colour in [(0, 0, 1), (1, 0, 0)]:
                page = doc.new_page(width=20, height=10)
                page.draw_rect(page.rect, color=colour, fill=colour)
            doc.save(pdf)
        w._load_path(str(pdf))
        w.upload.page_combo.setCurrentIndex(1)
        w.upload.dpi.setValue(300)
        self.idle()
        self.assertEqual(w._path, str(pdf))
        self.assertEqual(w._committed_pdf, (1, 300, 2))
        self.assertEqual(w.upload.current_page(), 1)
        self.assertEqual(w.upload.current_dpi(), 300)

    def test_error_from_old_settings_is_ignored_before_debounce_dispatch(self):
        w = self.window
        pipe = w._worker.pipeline
        run = pipe.run
        calls = []
        def fails_once(*args, **kwargs):
            if not calls:
                calls.append(True)
                time.sleep(.1)
                raise RuntimeError("obsolete settings failed")
            return run(*args, **kwargs)
        with patch.object(pipe, "run", side_effect=fails_once):
            with patch.object(QMessageBox, "warning") as warning:
                w.params = w.params.copy(board_title='obsolete')
                w._render_now()
                w.params = w.params.copy(board_title='current title')
                w._timer.setInterval(350)
                w._timer.start()
                self.idle()
                warning.assert_not_called()
        self.assertTrue(w._export_ready())

    def test_obsolete_render_and_stale_export_are_rejected(self):
        w = self.window
        pipe = w._worker.pipeline
        run = pipe.run
        def slow(*args, **kwargs):
            time.sleep(.15)
            return run(*args, **kwargs)
        with patch.object(pipe, "run", side_effect=slow):
            w.params = w.params.copy(board_title='obsolete')
            w._render_now()
            w.params = w.params.copy(board_title='current title')
            w._render_now()
            with patch.object(QFileDialog, "getSaveFileName") as dialog:
                w._export_svg()
                dialog.assert_not_called()
            self.assertFalse(w.act_svg.isEnabled())
            self.idle()
        self.assertTrue(w._export_ready())
        self.assertTrue(any(t.text == "current title" for layer in w._result.scene.layers for t in layer.texts))
        target = str(Path(self.tmp.name) / "current.svg")
        with patch.object(QFileDialog, "getSaveFileName", return_value=(target, "SVG")):
            w._export_svg()
        self.assertTrue(Path(target).is_file())



class ProjectReliability(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])
        cls.app.setQuitOnLastWindowClosed(False)

    def test_unit_change_emits_converted_dimensions_once(self):
        from studio.ui.controls import ControlsPanel
        from studio.params import Params
        controls = ControlsPanel(Params(card_w=5, paper_w=10, paper_h=7))
        states = []
        controls.changed.connect(lambda: states.append(controls.collect(Params())))
        controls.card_units.setCurrentIndex(controls.card_units.findData('cm'))
        self.assertEqual(len(states), 1)
        self.assertEqual(states[0].canvas_units, 'cm')
        self.assertEqual(states[0].card_w, 12.7)
        self.assertEqual(states[0].paper_w, 25.4)
        self.assertAlmostEqual(states[0].paper_h, 17.8)

    def test_notes_survive_navigation_and_failed_save_keeps_painting_open(self):
        from studio.library import Library, Swatch
        from studio.ui.easel_window import EaselWindow
        with tempfile.TemporaryDirectory() as root:
            lib = Library(root)
            a = lib.create_project('A', 'Oil — split primary 8')
            b = lib.create_project('B', 'Oil — split primary 8')
            sw = Swatch('s', '#888888', .5, .5, .1, None, a.palette, 'oil')
            lib.add_swatch(a, sw)
            win = EaselWindow(library=lib)
            try:
                win.show_project(a.id)
                win._show(win.project.swatch('s'))
                win.notes_edit.setPlainText('Remember this mix after it dries')
                win.project_combo.setCurrentIndex(win.project_combo.findData(b.id))
                self.assertEqual(lib.load_project(a.id).swatch('s').notes,
                                 'Remember this mix after it dries')
                win.show_project(a.id)
                win._show(win.project.swatch('s'))
                win.notes_edit.setPlainText('Unsaved because the disk failed')
                with patch.object(lib, 'save_project', side_effect=OSError('disk failure')):
                    with patch.object(QMessageBox, 'warning'):
                        win.project_combo.setCurrentIndex(win.project_combo.findData(b.id))
                self.assertEqual(win.project.id, a.id)
                self.assertEqual(win.project_combo.currentData(), a.id)
                self.assertEqual(win.notes_edit.toPlainText(), 'Unsaved because the disk failed')
            finally:
                win.close()

    def test_document_palette_canvas_and_stale_easel_save_remain_consistent(self):
        from studio.library import Library, Swatch
        from studio.models import Paint
        from studio.params import Params
        from studio.session import params_to_dict
        from studio.ui.easel_window import EaselWindow
        with tempfile.TemporaryDirectory() as root:
            lib = Library(root)
            p = lib.create_project('Study', 'Oil — split primary 8', canvas_w=10, canvas_h=7)
            stale = lib.load_project(p.id)
            paints = (Paint('My white', '#f6f4ef', True), Paint('My blue', '#0d4f8b'))
            settings = Params(palette=paints, paper_w=20, paper_h=14)
            lib.save_document(p.id, {'params': params_to_dict(settings)})
            lib.add_swatch(stale, Swatch('new', '#888888', .5, .5, .1, None, p.palette, 'oil'))
            back = lib.load_project(p.id)
            self.assertEqual((back.canvas_w, back.canvas_h), (20, 14))
            self.assertEqual(lib.palette(back.palette).paints, paints)
            self.assertIsNotNone(back.swatch('new'))
            win = EaselWindow(library=lib)
            try:
                win.show_project(p.id)
                self.assertEqual(win.palette.paints, paints)
                self.assertEqual((win.project.canvas_w, win.project.canvas_h), (20, 14))
                win.project.canvas_w = 30
                lib.save_project(win.project)
                self.assertEqual(lib.load_project(p.id).document['params']['paper_w'], 30)
            finally:
                win.close()

    def test_cross_process_saves_lock_the_entire_transaction(self):
        from studio.library import Library
        with tempfile.TemporaryDirectory() as root:
            lib = Library(root)
            proj = lib.create_project('Race', 'Oil — split primary 8')
            marker = Path(root) / 'writing'
            release = Path(root) / 'release'
            done = Path(root) / 'swatch-saved'
            writer = """
import sys,time
from pathlib import Path
from studio import library as module
lib=module.Library(sys.argv[1])
original=module._atomic_write
def paused(path,text):
    if path.endswith('project.json'):
        Path(sys.argv[1],'writing').touch()
        end=time.monotonic()+8
        while not Path(sys.argv[1],'release').exists():
            if time.monotonic()>end: raise RuntimeError('release timeout')
            time.sleep(.01)
    original(path,text)
module._atomic_write=paused
lib.save_document(sys.argv[2],{'params':{'board_title':'updated'}})
"""
            swatcher = """
import sys
from pathlib import Path
from studio.library import Library,Swatch
lib=Library(sys.argv[1])
proj=lib.load_project(sys.argv[2])
lib.add_swatch(proj,Swatch('concurrent','#888888',.5,.5,.2,None,proj.palette,'oil'))
Path(sys.argv[1],'swatch-saved').touch()
"""
            a = subprocess.Popen([sys.executable, '-c', writer, root, proj.id],
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            b = None
            try:
                wait_for(marker.exists)
                b = subprocess.Popen([sys.executable, '-c', swatcher, root, proj.id],
                                     stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                time.sleep(.3)
                self.assertFalse(done.exists(), 'second writer bypassed the transaction lock')
                release.touch()
                out, err = a.communicate(timeout=10)
                self.assertEqual(a.returncode, 0, out + err)
                out, err = b.communicate(timeout=10)
                self.assertEqual(b.returncode, 0, out + err)
                back = lib.load_project(proj.id)
                self.assertEqual(back.document['params']['board_title'], 'updated')
                self.assertIsNotNone(back.swatch('concurrent'))
            finally:
                release.touch()
                for proc in (a, b):
                    if proc is not None and proc.poll() is None:
                        proc.kill()
                        proc.communicate()

    def test_stale_document_preserves_shared_edits_and_rejects_conflicts(self):
        from copy import deepcopy
        from studio.library import Library
        from studio.params import Params
        from studio.session import params_to_dict
        with tempfile.TemporaryDirectory() as root:
            lib = Library(root)
            p = lib.create_project('Shared study', 'Oil — split primary 8')
            before = {'params': params_to_dict(Params(paper_w=10, paper_h=7))}
            lib.save_document(p.id, before)
            elsewhere = lib.load_project(p.id)
            elsewhere.palette = 'Acrylic — basic 6'
            elsewhere.canvas_w, elsewhere.canvas_h = 20, 14
            lib.save_project(elsewhere)
            changed = deepcopy(before)
            changed['params']['board_title'] = 'A document-only edit'
            back = lib.save_document(p.id, changed, previous_document=before)
            self.assertEqual((back.canvas_w, back.canvas_h), (20, 14))
            self.assertEqual(back.palette, 'Acrylic — basic 6')
            self.assertEqual(back.document['params']['board_title'], 'A document-only edit')
            conflict = deepcopy(before)
            conflict['params']['paper_w'] = 30
            with self.assertRaises(ValueError):
                lib.save_document(p.id, conflict, previous_document=before)
            self.assertEqual(lib.load_project(p.id).canvas_w, 20)

    def test_conflicting_swatch_delete_preserves_crop_and_current_notes(self):
        from studio.library import Library, Swatch
        with tempfile.TemporaryDirectory() as root:
            lib = Library(root)
            p = lib.create_project('Notes', 'Oil — split primary 8')
            lib.add_swatch(p, Swatch('s', '#888888', .5, .5, .1, None, p.palette, 'oil'),
                           np.full((4, 4, 3), 128, np.uint8))
            stale = lib.load_project(p.id)
            fresh = lib.load_project(p.id)
            fresh.swatch('s').notes = 'Newer observation'
            lib.save_project(fresh)
            with self.assertRaises(ValueError):
                lib.delete_swatch(stale, 's')
            back = lib.load_project(p.id)
            self.assertEqual(back.swatch('s').notes, 'Newer observation')
            self.assertTrue(Path(lib.crop_path(back, back.swatch('s'))).is_file())
            self.assertIsNotNone(stale.swatch('s'))


class ShutdownReliability(unittest.TestCase):
    def test_slow_processing_keeps_close_responsive_and_exits_safely(self):
        code = """
import os, sys, time, tempfile
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
os.environ['XDG_CONFIG_HOME'] = tempfile.mkdtemp()
os.environ['BARGUE_DATA_DIR'] = tempfile.mkdtemp()
os.environ['PAINTING_STUDIO_LIBRARY'] = tempfile.mkdtemp()
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QTimer
from studio.pipeline import Pipeline
from studio.ui.main_window import MainWindow
started = []
def slow(self, *args, **kwargs):
    started.append(True)
    time.sleep(6)
    raise RuntimeError('end of simulated long processing')
Pipeline.run = slow
app = QApplication([])
w = MainWindow()

w.show()
observations = []
QTimer.singleShot(250, w.close)
QTimer.singleShot(3500, lambda: observations.append(w.isVisible() and w._thread.isRunning()))
begin = time.monotonic()
assert app.exec() == 0
assert started and observations == [True]
assert time.monotonic() - begin >= 5.5
assert not w._thread.isRunning()
"""
        result = subprocess.run([sys.executable, "-c", code],
                                cwd=Path(__file__).resolve().parents[1],
                                capture_output=True, text=True, timeout=20)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertNotIn("Destroyed while thread", result.stderr)

if __name__ == "__main__":
    unittest.main()
