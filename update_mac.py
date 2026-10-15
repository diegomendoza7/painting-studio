"""Replace an existing Mac installation while preserving its launchers and data.

Run this script from the newly downloaded, extracted release:
    python3 /path/to/painting-studio-download/update_mac.py
"""
from __future__ import annotations

from datetime import datetime
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile

APP_FILES = ("studio", "main.py", "requirements.txt", "README.md")


def is_app(path: Path) -> bool:
    return (path / "main.py").is_file() and (path / "studio" / "app.py").is_file()


def find_installation(home: Path) -> Path:
    """Read launcher destinations without executing or rewriting their shell code."""
    targets = set()
    desktop = home / "Desktop"
    for launcher in desktop.rglob("*.command"):
        if len(launcher.relative_to(desktop).parts) > 4:
            continue
        try:
            lines = launcher.read_text().splitlines()
        except (OSError, UnicodeError):
            continue
        for line in lines:
            try:
                words = shlex.split(line, comments=True)
            except ValueError:
                continue
            if len(words) < 2 or words[0] != "cd":
                continue
            destination = Path(os.path.expandvars(os.path.expanduser(words[1])))
            if destination.is_absolute() and is_app(destination):
                targets.add(destination.resolve())
    if len(targets) != 1:
        detail = "\n".join(str(p) for p in sorted(targets)) or "No matching Desktop launcher found."
        raise ValueError("Could not choose a single installed Painting Studio folder.\n"
                         + detail + "\nNo app files were changed. Send this output for help.")
    return targets.pop()


def replace_app(source: Path, target: Path) -> Path:
    """Stage new code, retain the old code, and restore it if replacement fails."""
    if not is_app(source) or not is_app(target):
        raise ValueError("Both folders must contain Painting Studio's main.py and studio/app.py.")
    if source.resolve() == target.resolve():
        raise ValueError("Run update_mac.py from the newly downloaded folder, not the installed app.")
    if not all((source / name).exists() for name in APP_FILES):
        raise ValueError("The downloaded app is incomplete. No app files were changed.")
    backup = target.parent / (target.name + "-backup-" + datetime.now().strftime("%Y%m%d-%H%M%S-%f"))
    with tempfile.TemporaryDirectory(prefix=".painting-update-", dir=target.parent) as temporary:
        staged = Path(temporary)
        for name in APP_FILES:
            original = source / name
            if original.is_dir():
                shutil.copytree(original, staged / name, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
            else:
                shutil.copy2(original, staged / name)
        backup.mkdir()
        previous, installed = [], []
        try:
            for name in APP_FILES:
                destination = target / name
                if destination.exists():
                    destination.rename(backup / name)
                    previous.append(name)
                (staged / name).rename(destination)
                installed.append(name)
        except Exception:
            for name in reversed(installed):
                destination = target / name
                if destination.is_dir():
                    shutil.rmtree(destination)
                else:
                    destination.unlink()
            for name in previous:
                (backup / name).rename(target / name)
            raise
    return backup


def main() -> int:
    if sys.platform != "darwin":
        print("This updater is for macOS.", file=sys.stderr)
        return 1
    source = Path(__file__).resolve().parent
    try:
        target = find_installation(Path.home())
        if source == target:
            raise ValueError("Run this updater from the newly downloaded folder.")
        print(f"Updating Painting Studio at {target}", flush=True)
        # Install using the Python running this updater, just as for the initial
        # install. This happens before changing any installed app files.
        subprocess.run([sys.executable, "-m", "pip", "install", "-r",
                        str(source / "requirements.txt")], check=True)
        backup = replace_app(source, target)
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        print(f"Update did not complete: {exc}", file=sys.stderr)
        return 1
    print(f"Updated. Open your existing Desktop launcher as usual.\n"
          f"Old app code is backed up at: {backup}\n"
          "Your launchers, Python environment, and saved projects were preserved.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
