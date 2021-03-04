#pragma once
#include <cstddef>


namespace gr {
namespace kernels {
/**
 * @brief An interface that all "kernels" must implement
 *
 * @tparam InputType
 * @tparam OutputType
 */
template <class InputType, class OutputType>
struct kernel {
    /**
     * @brief A signature for when the input and output buffers are of different sizes and
     * types
     *
     * @param in_buffer
     * @param out_buffer
     * @param num_input_items
     * @param num_output_items
     */
    virtual void operator()(InputType* in_buffer,
                            OutputType* out_buffer,
                            size_t num_input_items,
                            size_t num_output_items) = 0;

    /**
     * @brief A signature for when the operation is done on the same type and
     *
     * @param in_buffer
     * @param out_buffer
     * @param num_items
     */
    virtual void
    operator()(InputType* in_buffer, OutputType* out_buffer, size_t num_items) = 0;

    /**
     * @brief A signature for in-place operations
     *
     * @param buffer
     * @param num_items
     */
    virtual void operator()(InputType* buffer, size_t num_items) = 0;

    virtual ~kernel() = default;
};
} // namespace kernels
} // namespace gr
