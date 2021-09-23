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

### Ubuntu 20.04 Example ###

The following assumes use of a prefix `/opt/newsched` (change if needed to desired prefix location):

1. Install dependencies
```bash
sudo apt install cmake g++ ninja-build libzmq3-dev libboost-program-options-dev doxygen libyaml-cpp-dev libfmt-dev libspdlog-dev libgtest-dev libqwt-qt5-dev 
sudo -H pip3 install meson
```
2. Install volk to prefix:
```bash
cd
git clone --recursive https://github.com/gnuradio/volk.git
cd volk
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release -DPYTHON_EXECUTABLE=/usr/bin/python3 -DCMAKE_INSTALL_PREFIX=/opt/newsched ../
make -j8
sudo make install
```
3. Build and install newsched
```bash
cd
git clone https://github.com/gnuradio/newsched.git
cd newsched
meson setup build --buildtype=debugoptimized --prefix=/opt/newsched --pkg-config-path=/opt/newsched/lib/pkgconfig/
cd build
ninja -j8
sudo ninja install
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
- volk
