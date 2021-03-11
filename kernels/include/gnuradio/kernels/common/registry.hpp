#pragma once
#include <gnuradio/kernels/cpu/kernel.hpp>
#include <map>

namespace gr {
namespace kernels {

class kernel_registry
{
public:
    void create_kernel_registration(kernel_interface* kernel,
                                    DeviceType device,
                                    DataType input_type,
                                    DataType output_type);

    void read_kernel_registration(kernel_interface* kernel,
                                  DeviceType device,
                                  DataType input_type,
                                  DataType output_type);

    void update_kernel_registration(kernel_interface* kernel,
                                    DeviceType device,
                                    DataType input_type,
                                    DataType output_type);

    void delete_kernel_registration(kernel_interface* kernel,
                                    DeviceType device,
                                    DataType input_type,
                                    DataType output_type);

private:
    std::map<std::string, kernel_interface*> _registry;
};

} // namespace kernels
} // namespace gr
