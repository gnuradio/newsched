#pragma once

#include <gnuradio/sync_block.hpp>

// This should live in some other folder as its dependencies should be 
// quite different. In fact, it should not depend on almost anything that's currently 
// in this code-base so that it can be easily included in an external library.
// That implies that kernels cannot use the std::vector<block_work_input> and std::vector<block_work_output> 
// as its function signature. It should be something like (int *in, int *out).

// Kernels must have information about its type, device, and name (functionality) so that it 
// can be matched with the correct block/port. It seems to me that the block class serves only the purpose
// of glue between ports and kernels.
namespace gr {
    namespace kernels {
        
        template<typename T>
        work_return_code_t copy_kernel(std::vector<block_work_input>& work_input, std::vector<block_work_output>& work_output){
                auto* iptr = (uint8_t*)work_input[0].items;
                auto* optr = (uint8_t*)work_output[0].items;
                memcpy(optr, iptr, sizeof(T)*work_output[0].n_items);
                work_output[0].n_produced = work_output[0].n_items;
                return work_return_code_t::WORK_OK;
        };
    }
}
                                    
namespace gr {
    namespace blocks {

    template <typename T>
    class copy : public sync_block
    {
    public:
        enum params : uint32_t { id_nports, num_params };
        typedef std::shared_ptr<copy> sptr;

        static sptr make(size_t itemsize)
        {
            auto ptr = std::make_shared<copy>(itemsize);

            ptr->add_port(untyped_port::make(
                "input", port_direction_t::INPUT, itemsize));

            ptr->add_port(untyped_port::make(
                "out", port_direction_t::OUTPUT, itemsize));


            return ptr;
        }

        copy() : sync_block("copy")
        {
            block_kernel = nullptr;
        }

        virtual work_return_code_t work(std::vector<block_work_input>& work_input,
                                        std::vector<block_work_output>& work_output)
        {
            return block_kernel(work_input, work_output);
        }

    private:
    };

    // Instead of template specializations, we're hoping to pick the kernel based on the port type, block name, and device
    // However, the port is currently instantiated after the block is and even if it's not
    // we do not have a way to enforce that the port should be instantiated first.
    // At the very least, this still allows for the separation of a kernel library
    // from a block/scheduler library.
    template<> copy<gr_complex>::copy() : sync_block("copy") { block_kernel = &gr::kernels::copy_kernel<gr_complex>; };
    template<> copy<float>::copy() : sync_block("copy") { block_kernel = &gr::kernels::copy_kernel<float>; };
    template<> copy<uint8_t>::copy() : sync_block("copy") { block_kernel = &gr::kernels::copy_kernel<uint8_t>; };
    template<> copy<uint16_t>::copy() : sync_block("copy") { block_kernel = &gr::kernels::copy_kernel<uint16_t>; };
    template<> copy<uint32_t>::copy() : sync_block("copy") { block_kernel = &gr::kernels::copy_kernel<uint32_t>; };

    } // namespace blocks
} // namespace gr
