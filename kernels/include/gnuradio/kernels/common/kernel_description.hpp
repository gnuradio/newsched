#pragma once
#include <cstddef>


namespace gr {
namespace kernels {

enum DeviceType { CPU, GPU };
enum DataType {
    uint8,
    uint16,
    uint32,
    int8,
    int16,
    int32,
    float32,
    float64,
    complex64,
    complex128
};

/**
 * @brief An interface for all kernels to implement
 *
 */
struct kernel_description {

    /**
     * @brief So all kernels will need meta-data like this so that it can be matched, at
     * run-time, with appropriate blocks/ports.
     */
    DeviceType device_;
    DataType input_type_;
    DataType output_type_;
};
} // namespace kernels
} // namespace gr
