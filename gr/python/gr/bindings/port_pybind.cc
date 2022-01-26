
#include <pybind11/complex.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

#include <gnuradio/port.hh>
// pydoc.h is automatically generated in the build directory
// #include <basic_block_pydoc.h>

void bind_port(py::module& m)
{
    using port_base = ::gr::port_base;
    using port_f = ::gr::port<float>;
    using port_c = ::gr::port<gr_complex>;
    using port_s = ::gr::port<int16_t>;
    using port_i = ::gr::port<int32_t>;
    using port_b = ::gr::port<uint8_t>;

    using message_port = ::gr::message_port;
    using untyped_port = ::gr::untyped_port;
    
    py::class_<port_base, std::shared_ptr<port_base>>(
        m, "port_base")
        .def("format_descriptor", &gr::port_base::format_descriptor)
        .def("dims", &gr::port_base::dims)
        ;

    py::class_<port_f, port_base, std::shared_ptr<port_f>>(
        m, "port_f")
        .def(py::init(&port_f::make),py::arg("name"), py::arg("direction"), py::arg("dims")=std::vector<size_t>{1}, py::arg("optional")=false, py::arg("multiplicity")=1)
        ;
    py::class_<port_c, port_base, std::shared_ptr<port_c>>(
        m, "port_c")
        .def(py::init(&port_c::make),py::arg("name"), py::arg("direction"), py::arg("dims")=std::vector<size_t>{1}, py::arg("optional")=false, py::arg("multiplicity")=1)
        ;
    py::class_<port_s, port_base, std::shared_ptr<port_s>>(
        m, "port_s")
        .def(py::init(&port_s::make),py::arg("name"), py::arg("direction"), py::arg("dims")=std::vector<size_t>{1}, py::arg("optional")=false, py::arg("multiplicity")=1)
        ;
    py::class_<port_i, port_base, std::shared_ptr<port_i>>(
        m, "port_i")
        .def(py::init(&port_i::make),py::arg("name"), py::arg("direction"), py::arg("dims")=std::vector<size_t>{1}, py::arg("optional")=false, py::arg("multiplicity")=1)
        ;
    py::class_<port_b, port_base, std::shared_ptr<port_b>>(
        m, "port_b")
        .def(py::init(&port_b::make),py::arg("name"), py::arg("direction"), py::arg("dims")=std::vector<size_t>{1}, py::arg("optional")=false, py::arg("multiplicity")=1)
        ;

    py::class_<message_port, port_base, std::shared_ptr<message_port>>(
        m, "message_port")
        ;

    py::class_<untyped_port, port_base, std::shared_ptr<untyped_port>>(
        m, "untyped_port")
        .def(py::init(&untyped_port::make),
             py::arg("name"), 
             py::arg("direction"), 
             py::arg("itemsize"),
             py::arg("optional")=false, 
             py::arg("multiplicity")=1)
        ;
        

    py::enum_<gr::port_direction_t>(m, "port_direction_t")
        .value("INPUT", gr::port_direction_t::INPUT) 
        .value("OUTPUT", gr::port_direction_t::OUTPUT)             
        .export_values();

    py::enum_<gr::port_type_t>(m, "port_type_t")
        .value("STREAM", gr::port_type_t::STREAM ) 
        .value("MESSAGE", gr::port_type_t::MESSAGE)             
        .export_values();

}
