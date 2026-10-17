"""Use the bundled camera SDK with the existing ROS/Python environment."""
from pathlib import Path
import platform
import sys


def use_bundled_realsense(*, package_share_dir=None):
    if sys.version_info[:2] != (3, 10) or platform.system() != 'Linux' or platform.machine() != 'x86_64':
        raise RuntimeError('Bundled D435i SDK requires Linux x86_64 / Python 3.10')
    if package_share_dir is None:
        source_share = Path(__file__).resolve().parents[1]
        if (source_share / 'third_party/realsense/python').is_dir():
            package_share_dir = source_share
        else:
            from ament_index_python.packages import get_package_share_path
            package_share_dir = get_package_share_path('bxi_example_py_elf3')
    sdk_path = Path(package_share_dir) / 'third_party/realsense/python'
    if not (sdk_path / 'pyrealsense2').is_dir():
        raise RuntimeError(f'Bundled D435i SDK missing: {sdk_path}; rebuild this package')
    if str(sdk_path) not in sys.path:
        sys.path.insert(0, str(sdk_path))
    return sdk_path
