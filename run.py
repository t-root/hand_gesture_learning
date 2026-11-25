"""Chạy server — làm việc từ thư mục này: `python run.py`.

Tự cài các gói trong requirements.txt còn thiếu trước khi khởi động.
TensorFlow là tuỳ chọn (có nhánh dự phòng cho LSTM): cài lỗi thì chỉ cảnh báo.
Đặt biến môi trường HGL_NO_INSTALL=1 để bỏ qua bước tự cài.
"""

import os
import re
import subprocess
import sys
from importlib import metadata
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OPTIONAL = {'tensorflow'}
# Gói cung cấp cùng module (mediapipe kéo theo opencv-contrib-python thay cho bản headless).
ALIASES = {
    'opencv-python-headless': ['opencv-python', 'opencv-contrib-python', 'opencv-contrib-python-headless'],
}
SKIP_FILE = ROOT / 'backend' / 'storage' / '.skip_optional'


def _parse_requirements(path: Path):
    for raw in path.read_text(encoding='utf-8').splitlines():
        line = raw.split('#', 1)[0].strip()
        if not line or line.startswith('-'):
            continue
        name = re.split(r'[\[<>=!~; ]', line, maxsplit=1)[0]
        yield name, line


def _is_installed(name: str, line: str) -> bool:
    installed = None
    for cand in [name, *ALIASES.get(name, [])]:
        try:
            installed = metadata.version(cand)
            break
        except metadata.PackageNotFoundError:
            continue
    if installed is None:
        return False
    if name in ALIASES:
        return True
    try:
        from packaging.requirements import Requirement

        req = Requirement(line)
        return not req.specifier or req.specifier.contains(installed, prereleases=True)
    except Exception:
        return True


def ensure_requirements() -> None:
    req_file = ROOT / 'requirements.txt'
    if os.environ.get('HGL_NO_INSTALL') == '1' or not req_file.is_file():
        return
    missing = [(n, l) for n, l in _parse_requirements(req_file) if not _is_installed(n, l)]
    skipped_names = set(SKIP_FILE.read_text().split()) if SKIP_FILE.is_file() else set()
    missing = [(n, l) for n, l in missing if n.lower() not in skipped_names]
    if not missing:
        return
    print('Missing packages: ' + ', '.join(n for n, _ in missing), flush=True)
    failed_required = []
    skipped = set(SKIP_FILE.read_text().split()) if SKIP_FILE.is_file() else set()
    for name, line in missing:
        if name.lower() in skipped:
            continue
        print(f'Installing {line} ...', flush=True)
        code = subprocess.call([sys.executable, '-m', 'pip', 'install', line])
        if code != 0:
            if name.lower() in OPTIONAL:
                print(f'Warning: could not install optional package {name}; continuing without it.', flush=True)
                skipped.add(name.lower())
                SKIP_FILE.parent.mkdir(parents=True, exist_ok=True)
                SKIP_FILE.write_text(' '.join(sorted(skipped)))
            else:
                failed_required.append(name)
    if failed_required:
        sys.exit('Could not install: ' + ', '.join(failed_required) + '. Close any running server and retry.')


def main() -> None:
    ensure_requirements()
    import uvicorn

    uvicorn.run('backend.web:app', host='127.0.0.1', port=8000, reload=False)


if __name__ == '__main__':
    main()


def __getattr__(name: str):
    # Giữ tương thích: `from run import app, STATE` vẫn dùng được.
    if name in ('app', 'STATE'):
        ensure_requirements()
        from backend.web import STATE, app

        return {'app': app, 'STATE': STATE}[name]
    raise AttributeError(name)
