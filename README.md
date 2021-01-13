# newsched – the final frontier #

A messaging-based implementation of GNU Radio scheduling.

This is a clean-slate approach to write a GNU Radio runtime that works for humans.

Its goal hence is not compatibility with current GNU Radio; we'll merge it into GNU Radio and add existing block wrappers as soon as a MVP works.

There are several design documents in the process of being cleaned up right now; will be added here later on.

## Building and Installation ##

newsched uses meson and ninja to manage the build process, which can be installed via pip and your package manager

```bash
pip install meson
cd newsched
meson setup build --buildtype=debugoptimized
cd build
ninja
```

### Installation ###

```bash
meson setup build --buildtype=debugoptimized --prefix=[INSTALL_PREFIX]
cd build
ninja install
```

## Dependencies ##

newsched uses C++17, and has the following dependencies

- meson
- boost (program_options)
- zmq
- doxygen
- fmt
- spdlog
- yaml-cpp
- gtest
