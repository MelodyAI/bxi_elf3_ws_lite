# Bundled PICO runtime

This directory contains the prebuilt x86_64/CPython-3.10 runtime used by the
optional PICO launch mode:

- `roboticsservice/`: the required XRoboToolkit PC service executable and
  shared libraries;
- `python/`: the `xrobotoolkit_sdk` CPython extension.

The runtime is selected automatically when `RGMT_REFERENCE_MODE = "pico"` and
`PICO_USE_BUNDLED_RUNTIME = True`. It is not portable to a different CPU
architecture or Python ABI. The sender still needs the normal Python runtime
packages (`numpy`, `scipy`, and `pyzmq`).
