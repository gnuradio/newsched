# Cross compile dependencies into sysroot

export XILINX_VIVADO=/tools/Xilinx/Vivado/2019.1
export PATH=/tools/Xilinx/SDK/2019.1/bin:/tools/Xilinx/SDK/2019.1/gnu/microblaze/lin/bin:/tools/Xilinx/SDK/2019.1/gnu/arm/lin/bin:/tools/Xilinx/SDK/2019.1/gnu/microblaze/linux_toolchain/lin64_le/bin:/tools/Xilinx/SDK/2019.1/gnu/aarch32/lin/gcc-arm-linux-gnueabi/bin:/tools/Xilinx/SDK/2019.1/gnu/aarch32/lin/gcc-arm-none-eabi/bin:/tools/Xilinx/SDK/2019.1/gnu/aarch64/lin/aarch64-linux/bin:/tools/Xilinx/SDK/2019.1/gnu/aarch64/lin/aarch64-none/bin:/tools/Xilinx/SDK/2019.1/gnu/armr5/lin/gcc-arm-none-eabi/bin:/tools/Xilinx/Vivado/2019.1/bin:$PATH

SYSROOT=/opt/staging
PKG_CONFIG_PATH=${SYSROOT}/usr/lib/pkgconfig
CC=arm-linux-gnueabihf-gcc
CPP=arm-linux-gnueabihf-c++
CFLAGS=--sysroot=${SYSROOT}
LIBS==--sysroot=${SYSROOT}

rm -rf build
mkdir build && cd build

# pushd .
# rm -rf volk
# git clone https://github.com/gnuradio/volk --recursive --branch v2.3.0
# cd volk && mkdir build && cd build &&
# CXXFLAGS=$CFLAGS cmake .. -DCMAKE_FIND_ROOT_PATH=$SYSROOT \
#           -DCMAKE_LIBRARY_PATH 
#           -DCMAKE_PREFIX_PATH= $SYSROOT \
#           -DCMAKE_CXX_COMPILER=$CPP -DCMAKE_C_COMPILER=$CC -DCMAKE_CROSSCOMPILING=1 \
#           -DCMAKE_INSTALL_PREFIX=../../staging/usr
# make
# make install
# popd

# exit

pushd .
rm -rf yaml-cpp
git clone https://github.com/jbeder/yaml-cpp
cd yaml-cpp && mkdir build && cd build &&
CXXFLAGS=$CFLAGS CFLAGS=$CFLAGS cmake .. -DCMAKE_FIND_ROOT_PATH=$SYSROOT \
          -DCMAKE_SHARED_LINKER_FLAGS="--sysroot=$SYSROOT" \
          -DCMAKE_MODULE_LINKER_FLAGS="--sysroot=$SYSROOT" \
          -DCMAKE_CXX_COMPILER=$CPP -DCMAKE_CROSSCOMPILING=1 \
          -DCMAKE_INSTALL_PREFIX=../../staging/usr \
          -DYAML_BUILD_SHARED_LIBS=ON
make -j
make install
popd

pushd .
rm -rf fmt
git clone https://github.com/fmtlib/fmt
cd fmt && mkdir build && cd build &&
CXXFLAGS=$CFLAGS CFLAGS=$CFLAGS cmake .. -DCMAKE_FIND_ROOT_PATH=$SYSROOT \
          -DCMAKE_SHARED_LINKER_FLAGS="--sysroot=$SYSROOT" \
          -DCMAKE_MODULE_LINKER_FLAGS="--sysroot=$SYSROOT" \
          -DCMAKE_CXX_COMPILER=$CPP -DCMAKE_CROSSCOMPILING=1 \
          -DCMAKE_INSTALL_PREFIX=../../staging/usr
make -j15
make install
popd


pushd .
rm -rf googletest
git clone https://github.com/google/googletest
cd googletest && mkdir build && cd build &&
CXXFLAGS=$CFLAGS CFLAGS=$CFLAGS cmake .. -DCMAKE_FIND_ROOT_PATH=$SYSROOT \
          -DCMAKE_SHARED_LINKER_FLAGS="--sysroot=$SYSROOT" \
          -DCMAKE_MODULE_LINKER_FLAGS="--sysroot=$SYSROOT" \
          -DCMAKE_CXX_COMPILER=$CPP -DCMAKE_CROSSCOMPILING=1 \
          -DCMAKE_INSTALL_PREFIX=../../staging/usr \
          -DBUILD_SHARED_LIBS=ON
make -j15
make install
popd



pushd .
rm -rf spdlog
git clone https://github.com/gabime/spdlog.git
cd spdlog && mkdir build && cd build &&
CXXFLAGS=$CFLAGS CFLAGS=$CFLAGS cmake .. -DCMAKE_FIND_ROOT_PATH=$SYSROOT \
          -DCMAKE_SHARED_LINKER_FLAGS="--sysroot=$SYSROOT" \
          -DCMAKE_MODULE_LINKER_FLAGS="--sysroot=$SYSROOT" \
          -DCMAKE_CXX_COMPILER=$CPP -DCMAKE_CROSSCOMPILING=1 \
          -DCMAKE_INSTALL_PREFIX=../../staging/usr \
          -DSPDLOG_BUILD_SHARED=ON \
          -DSPDLOG_BUILD_TESTS=OFF
make -j15
make install
popd



# pushd .
# # rm -rf flatbuffers
# # git clone https://github.com/google/flatbuffers
# # cd flatbuffers && mkdir build && cd build &&
# cd flatbuffers && git checkout tags/v2.0.0
# cd build &&
# CXXFLAGS=$CFLAGS CFLAGS=$CFLAGS cmake .. -DCMAKE_FIND_ROOT_PATH=$SYSROOT \
#           -DCMAKE_SHARED_LINKER_FLAGS="--sysroot=$SYSROOT" \
#           -DCMAKE_MODULE_LINKER_FLAGS="--sysroot=$SYSROOT" \
#           -DCMAKE_CXX_COMPILER=$CPP -DCMAKE_CROSSCOMPILING=1 \
#           -DCMAKE_INSTALL_PREFIX="$SYSROOT"_clean/usr \
#           -DFLATBUFFERS_BUILD_TESTS=OFF
# make -j15
# make install
# popd