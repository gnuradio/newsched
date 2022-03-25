# The Build System

We have chosen to use [Meson Build](https://mesonbuild.com/), which greatly simplifies build configuration and has convenient mechanisms for cross compiling

Take a look through the commit to see how simple it was to set up dependencies, and create build and installation targets. 

Meson can be installed with pip (ninja is used instead of make, so be sure that installed as well)

```bash
pip install meson
```

## Dependencies
The top level `meson.build` defines the dependencies as objects that can be used throughout the rest of the `meson.build` files

## Build Options
These are defined in `meson_options.txt`.  As the project evolves, more options will be added, but they can be easily accessed form the command line

Here, there is one option:
```python
option('enable_testing', type : 'boolean', value : true)
```

For example, to disable testing, add `-Denable_testing=false` to the `meson setup` command

## Usage

To build:
```bash
cd newsched
meson setup build --buildtype=debugoptimized --prefix=[INSTALL_PREFIX] --libdir=lib -D[option={true,false}]
cd build
ninja
```

To install:
```bash
ninja install
```

## Doxygen

Doxygen is configured in the docs/doxygen folder as per usual