#include <cuComplex.h>
#include <cuComplex.h>
#include <cuda.h>
#include <cuda_runtime.h>

namespace gr {
namespace blocks {
namespace multiply_const_cu {

template <typename T>
void exec_kernel(
    const T* in, T* out, T k, int grid_size, int block_size, cudaStream_t stream);

template <typename T>
void get_block_and_grid(int* minGrid, int* minBlock);

} // namespace multiply_const
} // namespace blocks
} // namespace gr