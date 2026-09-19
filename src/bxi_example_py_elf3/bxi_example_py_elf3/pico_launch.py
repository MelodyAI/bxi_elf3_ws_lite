"""Optional PICO service and pose-sender launch actions.

When PICO mode is selected, this module starts the bundled XRoboToolkit PC
service/runtime and the lightweight in-package pose sender as managed launch
actions, so both are stopped with the launch system. External service paths can
be selected from model_config.py for a different installation.
"""

from __future__ import annotations

import os
from pathlib import Path

from ament_index_python.packages import get_package_share_path
from launch.actions import ExecuteProcess, TimerAction

from bxi_example_py_elf3.model_config import (
    PICO_AUTO_START_SENDER,
    PICO_AUTO_START_SERVICE,
    PICO_SERVICE_ROOT,
    PICO_SERVICE_START_DELAY,
    PICO_SENDER_ENDPOINT,
    PICO_SENDER_FPS,
    PICO_SENDER_PYTHON,
    PICO_SENDER_PYTHONPATH,
    PICO_USE_BUNDLED_RUNTIME,
    RGMT_REFERENCE_MODE,
)


def _prepend_env_paths(*paths: Path, current: str = "") -> str:
    values = [str(path) for path in paths if str(path)]
    if current:
        values.append(current)
    return os.pathsep.join(values)


def make_pico_launch_actions():
    """Return launch-managed PICO service/sender processes."""
    if str(RGMT_REFERENCE_MODE).lower() != "pico":
        return []

    actions = []
    package_share = Path(get_package_share_path("bxi_example_py_elf3"))
    bundled_root = package_share / "third_party" / "pico"
    bundled_service_root = bundled_root / "roboticsservice"
    bundled_python_root = bundled_root / "python"
    if PICO_SERVICE_ROOT:
        service_root = Path(PICO_SERVICE_ROOT).expanduser()
    elif PICO_USE_BUNDLED_RUNTIME:
        service_root = bundled_service_root
    else:
        service_root = Path("/opt/apps/roboticsservice")

    if PICO_AUTO_START_SERVICE:
        service_binary = service_root / "RoboticsServiceProcess"
        service_env = {
            "LD_LIBRARY_PATH": _prepend_env_paths(
                service_root,
                service_root / "lib",
                service_root / "SDK" / "x64",
                current=os.environ.get("LD_LIBRARY_PATH", ""),
            ),
            "QT_PLUGIN_PATH": _prepend_env_paths(
                service_root / "plugins",
                current=os.environ.get("QT_PLUGIN_PATH", ""),
            ),
            "QT_QML_PATH": _prepend_env_paths(
                service_root / "qml",
                current=os.environ.get("QT_QML_PATH", ""),
            ),
        }
        actions.append(
            ExecuteProcess(
                cmd=[str(service_binary)],
                cwd=str(service_root),
                name="xr_robotics_service",
                output="screen",
                emulate_tty=True,
                additional_env=service_env,
            )
        )

    if PICO_AUTO_START_SENDER:
        # PYTHONPATH must contain the parent of the package directory so an
        # arbitrary configured interpreter can import bxi_example_py_elf3.
        package_parent = Path(__file__).resolve().parent.parent
        sender_env = {}
        sender_python_paths = [package_parent]
        if PICO_USE_BUNDLED_RUNTIME:
            sender_python_paths.append(bundled_python_root)
        sender_python_paths.extend(
            Path(item).expanduser()
            for item in PICO_SENDER_PYTHONPATH.split(os.pathsep)
            if item
        )
        pythonpath = _prepend_env_paths(
            *sender_python_paths,
            current=os.environ.get("PYTHONPATH", ""),
        )
        if pythonpath:
            sender_env["PYTHONPATH"] = pythonpath
        sender_env["LD_LIBRARY_PATH"] = _prepend_env_paths(
            service_root,
            service_root / "lib",
            service_root / "SDK" / "x64",
            current=os.environ.get("LD_LIBRARY_PATH", ""),
        )
        sender = ExecuteProcess(
            cmd=[
                PICO_SENDER_PYTHON,
                "-m",
                "bxi_example_py_elf3.pico_pose_sender",
                "--endpoint",
                PICO_SENDER_ENDPOINT,
                "--fps",
                str(PICO_SENDER_FPS),
            ],
            output="screen",
            emulate_tty=True,
            additional_env=sender_env,
        )
        delay = float(PICO_SERVICE_START_DELAY) if PICO_AUTO_START_SERVICE else 0.0
        actions.append(TimerAction(period=max(0.0, delay), actions=[sender]))

    return actions
