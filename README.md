# newsched – the final frontier #

![Make Test](https://img.shields.io/github/workflow/status/gnuradio/newsched/build%20and%20run%20tests?style=for-the-badge)
![Release](https://img.shields.io/github/v/release/gnuradio/newsched?style=for-the-badge)
[![AUR](https://img.shields.io/github/license/gnuradio/newsched?style=for-the-badge)](https://github.com/gnuradio/newsched/blob/main/COPYING)

<table><tr>
<th><b><a href="https://gnuradio.github.io/newsched/">Documentation</a></b></th>
</tr></table>

A messaging-based implementation of GNU Radio scheduling.

This is a clean-slate approach to write a GNU Radio runtime that works for humans.

Its goal hence is not compatibility with current GNU Radio; we'll merge it into GNU Radio and add existing block wrappers as soon as a MVP works.

There are several design documents in the process of being cleaned up right now; will be added here later on.

## Building and Installation ##

For build and installation info, and instructions for Ubuntu 20.04, see the [Getting Started Guide](https://gnuradio.github.io/newsched/user_tutorial/02_Getting)

## Dependencies ##

newsched uses C++17, and has the following dependencies

- meson
- boost
- zmq
- doxygen
- fmt
- spdlog
- yaml-cpp
- gtest
- volk
