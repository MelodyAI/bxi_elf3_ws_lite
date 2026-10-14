# Bundled PICO runtime

This directory contains the prebuilt x86_64/CPython-3.10 runtime used by the
optional PICO launch mode:

- `roboticsservice/`: the required XRoboToolkit PC service executable and
  shared libraries;
- `python/`: the `xrobotoolkit_sdk` CPython extension.

The runtime is selected automatically when `RGMT_REFERENCE_MODE` is `"pico"` or `"pico_umr"` and
`PICO_USE_BUNDLED_RUNTIME = True`. It is not portable to a different CPU
architecture or Python ABI. The sender still needs the normal Python runtime
packages (`numpy`, `scipy`, and `pyzmq`).

`pico` publishes the existing `neural14` contract for neural_retarget.
`pico_umr` publishes `smpl22` for live UMR, including spine, collar, neck, head
and foot nodes. Launch adds the matching sender `--body-layout` automatically.
When starting the sender manually, use `--body-layout smpl22` for `pico_umr`.
