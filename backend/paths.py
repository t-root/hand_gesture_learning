"""Đường dẫn trong gói backend và thư mục dữ liệu."""

import shutil
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
ROOT_DIR = APP_DIR.parent
FRONTEND_DIR = ROOT_DIR / 'frontend'
STATIC_INDEX = FRONTEND_DIR / 'index.html'

STORAGE_DIR = APP_DIR / 'storage'
EXPORT_DIR = APP_DIR / 'exports'


def _ensure_frontend_index() -> None:
    FRONTEND_DIR.mkdir(parents=True, exist_ok=True)
    if STATIC_INDEX.is_file():
        return
    for src in (APP_DIR / 'index.html', ROOT_DIR / 'index.html'):
        if not src.is_file():
            continue
        try:
            src.rename(STATIC_INDEX)
        except OSError:
            shutil.copy2(src, STATIC_INDEX)
        break


_ensure_frontend_index()


def _populate_dir(dest: Path, *candidates: Path) -> None:
    if dest.exists():
        return
    for src in candidates:
        if not src.is_dir():
            continue
        try:
            src.rename(dest)
        except OSError:
            try:
                shutil.copytree(src, dest)
            except OSError:
                continue
        break


# Layout cũ: `storage` / `exports` nằm cùng cấp thư mục `backend/`
_populate_dir(STORAGE_DIR, ROOT_DIR / 'storage')
_populate_dir(EXPORT_DIR, ROOT_DIR / 'exports', ROOT_DIR / 'server_exports')
_populate_dir(EXPORT_DIR, APP_DIR / 'server_exports')

STORAGE_DIR.mkdir(exist_ok=True)
EXPORT_DIR.mkdir(exist_ok=True)
