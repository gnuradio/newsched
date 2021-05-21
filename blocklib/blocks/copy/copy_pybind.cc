#include <pybind11/complex.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

#include <gnuradio/blocks/copy.hh>
// pydoc.h is automatically generated in the build directory
// #include <copy_pydoc.h>

void bind_copy(py::module& m)
{
    using copy = ::gr::blocks::copy;

    py::class_<copy, gr::sync_block, gr::block, gr::node, std::shared_ptr<copy>> copy_class(m, "copy");

    py::enum_<::gr::blocks::copy::available_impl>(copy_class, "available_impl")
        .value("cpu", ::gr::blocks::copy::available_impl::CPU)   // 0
        .value("cuda", ::gr::blocks::copy::available_impl::CUDA) // 1
        .export_values();

    copy_class.def(py::init([](size_t itemsize, gr::blocks::copy::available_impl impl) {
                       return copy::make({ itemsize }, impl);
                   }),
                   py::arg("itemsize"),
                   py::arg("impl") = gr::blocks::copy::available_impl::CPU);
}
