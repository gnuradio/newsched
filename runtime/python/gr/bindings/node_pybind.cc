
#include <pybind11/complex.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

#include <gnuradio/node.hh>
// pydoc.h is automatically generated in the build directory
// #include <basic_block_pydoc.h>

void bind_node(py::module& m)
{
    using node = ::gr::node;

    py::class_<node,  std::shared_ptr<node>>(
        m, "node")

        .def("name", &node::name)
        .def("alias", &node::alias)
        .def("input_port", &node::input_port)
        .def("output_port", &node::output_port)
        ;
}
