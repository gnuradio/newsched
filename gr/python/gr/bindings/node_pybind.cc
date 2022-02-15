
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
        .def("get_port", py::overload_cast<unsigned int, gr::port_type_t, gr::port_direction_t>(&gr::node::get_port))
        .def("get_message_port", &node::get_message_port)
        ;
}
