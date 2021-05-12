# Building newsched for PLUTOSDR

- Install Xilinx SDK
- Install plutosdr sysroot
- run build_deps.sh
- cross compile newsched

```
meson setup ../ build_pluto --cross-file cross_compile.ini -Denable_cuda=false -Denable_testing=true --prefix=$(readlink -f ./build/staging/usr)
cd build_pluto
ninja
ninja install
```