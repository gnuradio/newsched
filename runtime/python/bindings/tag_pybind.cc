/*
 * Copyright 2020 Free Software Foundation, Inc.
 *
 * This file is part of GNU Radio
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 */

#include <pybind11/complex.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

#include <gnuradio/tag.hh>
// pydoc.h is automatically generated in the build directory
// #include <block_pydoc.h>

void bind_tag(py::module& m)
{
    py::enum_<gr::tag_propagation_policy_t>(m, "tag_propagation_policy_t")
        .value("TPP_DONT", gr::tag_propagation_policy_t::TPP_DONT)             // 0
        .value("TPP_ALL_TO_ALL", gr::tag_propagation_policy_t::TPP_ALL_TO_ALL) // 1
        .value("TPP_ONE_TO_ONE", gr::tag_propagation_policy_t::TPP_ONE_TO_ONE) // 2
        .value("TPP_CUSTOM", gr::tag_propagation_policy_t::TPP_CUSTOM)         // 3
        .export_values();

}
