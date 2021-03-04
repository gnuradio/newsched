#include <gnuradio/kernels/cpu/copy.hpp>
#include <cstring>

namespace gr {
    namespace kernels {
        void copy_kernel(uint8_t* in_buffer, size_t num_input_items, uint8_t* out_buffer, size_t num_output_items){
                memcpy(out_buffer, in_buffer, num_input_items);
        };
    }
}

