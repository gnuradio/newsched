#include <pybind11/complex.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

#include <gnuradio/blocks/vector_source.hh>
// pydoc.h is automatically generated in the build directory
// #include <vector_source_pydoc.h>

template <typename T>
void bind_vector_source_template(py::module& m, const char* classname)
{
    py::class_<gr::blocks::vector_source<T>,
               gr::sync_block,
               gr::block,
               gr::node,
               std::shared_ptr<gr::blocks::vector_source<T>>>
        vector_source_class(m, classname);

    py::enum_<typename gr::blocks::vector_source<T>::available_impl>(vector_source_class,
                                                                     "available_impl")
        .value("cpu", ::gr::blocks::vector_source<T>::available_impl::CPU) // 0
        // .value("cuda", ::gr::blocks::vector_source::available_impl::CUDA) // 1
        .export_values();

    vector_source_class.def(
        py::init([](const std::vector<T>& data,
                    bool repeat,
                    unsigned int vlen,
                    const std::vector<gr::tag_t>& tags,
                    typename gr::blocks::vector_source<T>::available_impl impl) {
            return gr::blocks::vector_source<T>::make({ data, repeat, vlen, tags }, impl);
        }),
        py::arg("data"),
        py::arg("repeat") = false,
        py::arg("vlen") = 1,
        py::arg("tags") = std::vector<gr::tag_t>(),
        py::arg("impl") = gr::blocks::vector_source<T>::available_impl::CPU);
}


void bind_vector_source(py::module& m)
{
    bind_vector_source_template<std::uint8_t>(m, "vector_source_b");
    bind_vector_source_template<std::int16_t>(m, "vector_source_s");
    bind_vector_source_template<std::int32_t>(m, "vector_source_i");
    bind_vector_source_template<float>(m, "vector_source_f");
    bind_vector_source_template<gr_complex>(m, "vector_source_c");
}
