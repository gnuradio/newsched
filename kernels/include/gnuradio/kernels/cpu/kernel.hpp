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
struct kernel_interface {

    /**
     * @brief So all kernels will need meta-data like this so that it can be matched, at
     * run-time, with appropriate blocks/ports.
     *
     */
    DeviceType device_;
    DataType input_type_;
    DataType output_type_;


    /**
     * @brief A signature for when the input and output buffers are of different sizes
     * @param in_buffer
     * @param out_buffer
     * @param num_input_items
     * @param num_output_items
     */
    virtual void operator()(void* in_buffer,
                            void* out_buffer,
                            size_t num_input_items,
                            size_t num_output_items) = 0;

    /**
     * @brief A signature for when the input and output buffers are different, but of the
     * same size
     *
     * @param in_buffer
     * @param out_buffer
     * @param num_items
     */
    virtual void operator()(void* in_buffer, void* out_buffer, size_t num_items) = 0;

    /**
     * @brief A signature for in-place operations
     *
     * @param buffer
     * @param num_items
     */
    virtual void operator()(void* buffer, size_t num_items) = 0;

    /**
     * @brief Destroy the kernel interface object
     *
     */
    virtual ~kernel_interface() = default;
};
} // namespace kernels
} // namespace gr
