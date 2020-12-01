#include <gnuradio/blocklib/cuda/copy.hpp>

#include <cuComplex.h>
#include "helper_cuda.h"

extern void
apply_copy(cuFloatComplex* in, cuFloatComplex* out, int grid_size, int block_size);
extern void get_block_and_grid(int* minGrid, int* minBlock);

namespace gr {
namespace cuda {
/*
 * The private constructor
 */
copy::copy(const size_t batch_size) : gr::sync_block("copy"), d_batch_size(batch_size)

{
    get_block_and_grid(&d_min_grid_size, &d_block_size);
    std::cout << "minGrid: " << d_min_grid_size << ", blockSize: " << d_block_size
              << std::endl;

    if (batch_size < d_block_size) {
        throw std::runtime_error("batch_size must be a multiple of block size");
    }
}

/*
 * Our virtual destructor.
 */
copy::~copy() {}

work_return_code_t copy::work(std::vector<block_work_input>& work_input,
                             std::vector<block_work_output>& work_output)
{
    const gr_complex* in = reinterpret_cast<const gr_complex*>(work_input[0].items);
    gr_complex* out = reinterpret_cast<gr_complex*>(work_output[0].items);

    auto noutput_items = work_output[0].n_items;
    auto mem_size = d_batch_size * sizeof(gr_complex);


    for (auto s = 0; s < noutput_items; s++) {

        auto in_data = const_cast<cuFloatComplex*>(
            reinterpret_cast<const cuFloatComplex*>(in + s * d_batch_size));
        auto out_data = reinterpret_cast<cuFloatComplex*>(out + s * d_batch_size);

        apply_copy(in_data, out_data, d_batch_size / d_block_size, d_block_size);

        cudaDeviceSynchronize();
    }

    // Tell runtime system how many output items we produced.
    work_output[0].n_produced = noutput_items;
    return work_return_code_t::WORK_OK;
}
} // namespace cuda
} // namespace gr
