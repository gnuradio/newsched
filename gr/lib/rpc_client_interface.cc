#include <gnuradio/rpc_client_interface.h>

#include <gnuradio/pyblock_detail.h>

#include <pybind11/embed.h>
#include <pybind11/pybind11.h> // must be first
#include <pybind11/stl.h>
namespace py = pybind11;

namespace gr {

void rpc_client_interface::set_pyblock_detail(std::shared_ptr<pyblock_detail> p)
{
    d_pyblock_detail = p;
}
std::shared_ptr<pyblock_detail> rpc_client_interface::pb_detail()
{
    return d_pyblock_detail;
}


void rpc_client_interface::block_method(const std::string& block_name,
                                        const std::string& method,
                                        const std::string& payload)
{
    if (pb_detail()) {
        py::gil_scoped_acquire acquire;

        py::object ret =
            this->pb_detail()->handle().attr("block_method")(block_name, method, payload);
    }
}

std::string rpc_client_interface::block_parameter_query(const std::string& block_name,
                                                      const std::string& parameter)
{
    if (pb_detail()) {
        py::gil_scoped_acquire acquire;

        py::object ret = this->pb_detail()->handle().attr("block_parameter_query")(
            block_name, parameter);

        return ret.cast<std::string>();
    }
}

void rpc_client_interface::block_parameter_change(const std::string& block_name,
                                                  const std::string& parameter,
                                                  const std::string& payload)
{
    if (pb_detail()) {
        py::gil_scoped_acquire acquire;

        py::object ret = this->pb_detail()->handle().attr("block_parameter_change")(
            block_name, parameter, payload);
    }
}

} // namespace gr