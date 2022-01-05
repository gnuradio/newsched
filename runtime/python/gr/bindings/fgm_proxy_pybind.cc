
/*
 * fgm_proxyright 2020 Free Software Foundation, Inc.
 *
 * This file is part of GNU Radio
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 */

#include <pybind11/complex.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <thread>

namespace py = pybind11;

#include <gnuradio/fgm_proxy.hh>
// pydoc.h is automatically generated in the build directory
// #include <block_pydoc.h>

void bind_fgm_proxy(py::module& m)
{
    py::class_<gr::fgm_proxy, std::shared_ptr<gr::fgm_proxy>>(
        m, "fgm_proxy")

        .def(py::init(&gr::fgm_proxy::make));
}
