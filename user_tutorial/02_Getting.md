# Getting and building newsched

The following instructions describe how to create a prefix and install newsched into it

## 1. Set up a prefix

```sh
PREFIX=/path/to/my/prefix
mkdir -p $PREFIX/src 
```
Put the following contents into a file called `setup_env.sh` (and adjust python paths as necessary) - (This is copied from PyBOMBS)
```bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
export PATH="$DIR/bin:$PATH"
export PYTHONPATH="$DIR/lib/python3/site-packages:$DIR/lib/python3/dist-packages:$DIR/lib/python3.8/site-packages:$DIR/lib/python3.8/dist-packages:$DIR/lib64/python3/site-packages:$DIR/lib64/python3/dist-packages:$DIR/lib64/python3.8/site-packages:$DIR/lib64/python3.8/dist-packages:$PYTHONPATH"
export LD_LIBRARY_PATH="$DIR/lib:$DIR/lib64/:$DIR/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH"
export LIBRARY_PATH="$DIR/lib:$DIR/lib64/:$DIR/lib/x86_64-linux-gnu:$LIBRARY_PATH"
export PKG_CONFIG_PATH="$DIR/lib/pkgconfig:$DIR/lib64/pkgconfig:$PKG_CONFIG_PATH"
```

Every time a new terminal is launched where newsched is to be used, just call
```
source setup_env.sh
```

## 2. Install Dependencies

The following dependencies are necessary to build and run `newsched`

- [Meson Build](https://mesonbuild.com)
- [Ninja Build](https://ninja-build.org/)
- [VOLK](https://www.libvolk.org/doxygen/)
- [spdlog](https://github.com/gabime/spdlog)
- [yamlcpp](https://github.com/jbeder/yaml-cpp)
- [GTest](https://github.com/google/googletest)
- [Boost](https://www.boost.org/)
- FFTW
- C++17 capable compiler (g++9)
  
Optionally
- Python3
- Numpy
- QT/QWT
- SoapySDR

Required for building VOLK
- CMake
- python3-mako

On Ubuntu 20.04, the following should be sufficient to get all the prerequisites installed
```bash
sudo apt-get update -q  && apt-get -y upgrade
sudo apt-get install -qy \
    build-essential \
    --no-install-recommends \
    libspdlog-dev \
    libyaml-cpp-dev \
    libgtest-dev \
    libfmt-dev \
    pybind11-dev \
    python3-dev \
    python3-numpy \
    libqwt-qt5-dev \
    ninja-build \
    libboost-dev \
    libboost-program-options-dev \
    libboost-thread-dev \
    libfftw3-dev \
    git \ 
    cmake \
    pkg-config \
    python3-pip

pip3 install meson mako
```

VOLK must be built from source
```bash
cd $PREFIX/src
git clone --recursive https://github.com/gnuradio/volk.git --branch v2.4.1
cd volk && mkdir build && cd build &&cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/prefix && make install -j8
```

## 3. Clone and build newsched
```bash
cd $PREFIX/src
git clone https://github.com/gnuradio/newsched
cd newsched
meson setup build --buildtype=debugoptimized --prefix=$PREFIX --libdir=lib
cd build
ninja
ninja test
ninja install
```

## Installing from Packaging

TODO: build and distribute conda and ppa packages