#include <pybind11/complex.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

#include <gnuradio/blocks/vector_sink.hh>
// pydoc.h is automatically generated in the build directory
// #include <vector_sink_pydoc.h>

template <typename T>
void bind_vector_sink_template(py::module& m, const char* classname)
{
    py::class_<gr::blocks::vector_sink<T>,
               gr::sync_block,
               gr::block,
               gr::node,
               std::shared_ptr<gr::blocks::vector_sink<T>>>
        vector_sink_class(m, classname);

    py::enum_<typename gr::blocks::vector_sink<T>::available_impl>(vector_sink_class,
                                                                     "available_impl")
        .value("cpu", ::gr::blocks::vector_sink<T>::available_impl::CPU) // 0
        // .value("cuda", ::gr::blocks::vector_sink::available_impl::CUDA) // 1
        .export_values();

    vector_sink_class.def(
        py::init([](const std::vector<T>& data,
                    bool repeat,
                    unsigned int vlen,
                    const std::vector<gr::tag_t>& tags,
                    typename gr::blocks::vector_sink<T>::available_impl impl) {
            return gr::blocks::vector_sink<T>::make({ data, repeat, vlen, tags }, impl);
        }),
        py::arg("vlen") = 1,
        py::arg("reserve_items") = 1024,
        py::arg("impl") = gr::blocks::vector_sink<T>::available_impl::CPU);
}


void bind_vector_sink(py::module& m)
{
    bind_vector_sink_template<std::uint8_t>(m, "vector_sink_b");
    bind_vector_sink_template<std::int16_t>(m, "vector_sink_s");
    bind_vector_sink_template<std::int32_t>(m, "vector_sink_i");
    bind_vector_sink_template<float>(m, "vector_sink_f");
    bind_vector_sink_template<gr_complex>(m, "vector_sink_c");
}
