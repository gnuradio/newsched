#pragma once

#include <gnuradio/sync_block.hpp>
#include <gnuradio/kernels/cpu/copy.hpp>
                                    
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
    // template<> copy<gr_complex>::copy() : sync_block("copy") { block_kernel = &gr::kernels::copy_kernel<gr_complex>; };
    // template<> copy<float>::copy() : sync_block("copy") { block_kernel = &gr::kernels::copy_kernel<float>; };
    // template<> copy<uint8_t>::copy() : sync_block("copy") { block_kernel = &gr::kernels::copy_kernel<uint8_t>; };
    // template<> copy<uint16_t>::copy() : sync_block("copy") { block_kernel = &gr::kernels::copy_kernel<uint16_t>; };
    // template<> copy<uint32_t>::copy() : sync_block("copy") { block_kernel = &gr::kernels::copy_kernel<uint32_t>; };

    } // namespace blocks
} // namespace gr
