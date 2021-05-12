# Cross compile dependencies into sysroot

export XILINX_VIVADO=/tools/Xilinx/Vivado/2019.1
export PATH=/tools/Xilinx/SDK/2019.1/bin:/tools/Xilinx/SDK/2019.1/gnu/microblaze/lin/bin:/tools/Xilinx/SDK/2019.1/gnu/arm/lin/bin:/tools/Xilinx/SDK/2019.1/gnu/microblaze/linux_toolchain/lin64_le/bin:/tools/Xilinx/SDK/2019.1/gnu/aarch32/lin/gcc-arm-linux-gnueabi/bin:/tools/Xilinx/SDK/2019.1/gnu/aarch32/lin/gcc-arm-none-eabi/bin:/tools/Xilinx/SDK/2019.1/gnu/aarch64/lin/aarch64-linux/bin:/tools/Xilinx/SDK/2019.1/gnu/aarch64/lin/aarch64-none/bin:/tools/Xilinx/SDK/2019.1/gnu/armr5/lin/gcc-arm-none-eabi/bin:/tools/Xilinx/Vivado/2019.1/bin:$PATH

export DIRECTORY=build_pluto

# If the staging directory doesn't exist, create it
if [[ ! -d "./build/staging" ]]
then
    mkdir -p ./build/staging
fi


if [[ ! -d "$DIRECTORY" ]]
then
    meson setup ../../ build_pluto --cross-file cross_compile.ini -Denable_cuda=false -Denable_testing=true --prefix=$(readlink -f ./build/staging/usr)
fi

cd build_pluto
ninja install