"""Build and stage the KV metadata CANN vendor for the legacy wheel."""

import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "python/sgl_kernel_npu"))

from build_tools.cann import _remove_tree, build_cann


def main():
    arch, build_dir, destination = sys.argv[1:]
    build_dir = Path(build_dir)
    destination = Path(destination)
    try:
        result = build_cann(arch, ("kv_sparse_metadata",), build_root=build_dir)
        _remove_tree(destination)
        shutil.copytree(result.vendor, destination)
    finally:
        # CANN can install read-only directories. Clear the temporary install
        # so build.sh can still clean its build directory on the next run.
        _remove_tree(build_dir / "cann/installed")


if __name__ == "__main__":
    main()
