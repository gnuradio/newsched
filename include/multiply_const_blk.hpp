#ifndef INCLUDED_MULTIPLY_CONST_HPP
#define INCLUDED_MULTIPLY_CONST_HPP

#include "sync_block.hpp"

namespace gr {
namespace blocks {
template <class T>
class multiply_const : virtual public sync_block
{
private:
    /*   Get rid of this probably  */
    // Publish what used to be in the input signature -- does this go into the
    // This is all very ugly right now - punt for now on publishing this information
    static const io_signature_capability d_input_signature_capability(1, 1);
    static const io_signature_capability d_output_signature_capability(1, 1);
    /* ------------------------   */

    T d_k;
    const size_t d_vlen;


public:
    multiply_const(T k, size_t vlen = 1);
    ~multiply_const();

    virtual work_return_code_t work(std::vector<block_work_input>& work_input,
                                    std::vector<block_work_output>& work_output);

    const T k();
    void set_k(T k);
};

typedef multiply_const<std::int16_t> multiply_const_ss;
typedef multiply_const<std::int32_t> multiply_const_ii;
typedef multiply_const<float> multiply_const_ff;
typedef multiply_const<gr_complex> multiply_const_cc;

} // namespace blocks
} // namespace gr
#endif