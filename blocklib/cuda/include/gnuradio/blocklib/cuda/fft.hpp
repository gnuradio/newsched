#pragma once

#include <gnuradio/sync_block.hpp>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cufft.h>

namespace gr {
namespace cuda {

class fft : public sync_block
{
public:
    enum params : uint32_t { num_params };

    typedef std::shared_ptr<fft> sptr;
    static sptr make(const size_t fft_size,
                     const bool forward,
                     bool shift = false,
                     const size_t batch_size = 1)
    {
        auto ptr =
            std::make_shared<fft>(fft_size, forward, shift, batch_size);

        ptr->add_port(port<gr_complex>::make(
            "input", port_direction_t::INPUT, port_type_t::STREAM, {fft_size}));
        ptr->add_port(port<gr_complex>::make(
            "output", port_direction_t::OUTPUT, port_type_t::STREAM, {fft_size}));

        // ptr->add_param(
        //     param<T>::make(fft<T>::params::id_k, "k", k, &ptr->d_k));

        // // TODO: vlen should be const and unchangeable as a parameter
        // ptr->add_param(param<size_t>::make(
        //     fft<T>::params::id_vlen, "vlen", vlen, &ptr->d_vlen));

        return ptr;
    }

    fft(const size_t fft_size,
        const bool forward,
        bool shift,
        const size_t batch_size);
    ~fft();


    virtual work_return_code_t work(std::vector<block_work_input>& work_input,
                                    std::vector<block_work_output>& work_output);

private:
    size_t d_fft_size;
    bool d_forward;
    bool d_shift;
    size_t d_batch_size;

    cufftHandle d_plan = 0;

};

} // namespace cuda
} // namespace gr