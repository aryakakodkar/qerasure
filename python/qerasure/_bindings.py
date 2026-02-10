"""Loading helpers for the compiled qerasure Python extension."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path


_DEF_MODULE_NAME = "qerasure_python"


def _candidate_build_dirs() -> list[Path]:
    pkg_dir = Path(__file__).resolve().parent
    repo_root = pkg_dir.parent.parent
    names = [
        "build-release",
        "build",
        "build-clean",
        "cmake-build-release",
        "cmake-build-debug",
    ]
    candidates = [repo_root / name for name in names]
    candidates.extend(path for path in repo_root.glob("build*") if path.is_dir())
    return candidates


def _inject_extension_path() -> None:
    for build_dir in _candidate_build_dirs():
        if not build_dir.exists():
            continue
        module_candidates = list(build_dir.glob("qerasure_python*.so"))
        module_candidates.extend(build_dir.glob("qerasure_python*.pyd"))
        if module_candidates:
            build_dir_str = str(build_dir)
            if build_dir_str not in sys.path:
                sys.path.insert(0, build_dir_str)
            return


def load_cpp_module():
    try:
        return importlib.import_module(_DEF_MODULE_NAME)
    except ImportError:
        _inject_extension_path()
        return importlib.import_module(_DEF_MODULE_NAME)


cpp = load_cpp_module()
