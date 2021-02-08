#pragma once

#include <gnuradio/sync_block.hpp>

#include <cuda.h>
#include <cuda_runtime_api.h>

namespace gr {
namespace cuda {

class copy : public sync_block
{
public:
    enum params : uint32_t { num_params };

    typedef std::shared_ptr<copy> sptr;
    static sptr make(const size_t batch_size = 1, const size_t load = 1)
    {
        auto ptr = std::make_shared<copy>(batch_size, load);

        ptr->add_port(port<gr_complex>::make(
            "input", port_direction_t::INPUT, port_type_t::STREAM, { batch_size }));
        ptr->add_port(port<gr_complex>::make(
            "output", port_direction_t::OUTPUT, port_type_t::STREAM, { batch_size }));

        return ptr;
    }

    copy(const size_t batch_size, const size_t load);
    ~copy();


    virtual work_return_code_t work(std::vector<block_work_input>& work_input,
                                    std::vector<block_work_output>& work_output);

private:
    size_t d_batch_size;
    size_t d_load;
    int d_block_size;
    int d_min_grid_size;
    cudaStream_t stream;
};

} // namespace cuda
} // namespace gr