"""Application entry point."""
from __future__ import annotations

import sys


def main() -> int:
    from PySide6.QtWidgets import QApplication

    app = QApplication(sys.argv)
    app.setApplicationName("Painting Studio")
    app.setOrganizationName("Studio")
    if "--easel" in sys.argv[1:]:
        # Straight to the easel — the window you keep open while painting.
        from .ui.easel_window import EaselWindow

        window = EaselWindow()
    else:
        from .ui.main_window import MainWindow

        window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
