#pragma once

#include <gnuradio/sync_block.hpp>
#include <pmt/pmtf_scalar.hpp>

namespace gr {
namespace blocks {

template <class T>
class multiply_const : public sync_block
{
public:
    enum params : uint32_t { id_k, id_vlen, num_params };

    typedef std::shared_ptr<multiply_const> sptr;
    static sptr make(const T k, const size_t vlen = 1)
    {
        return std::make_shared<multiply_const>(multiply_const<T>(k, vlen));
    }
    multiply_const(T k, size_t vlen);

    virtual work_return_code_t work(std::vector<block_work_input>& work_input,
                                    std::vector<block_work_output>& work_output);

    // TODO: would like to NOT use macros - just don't know just what pattern to use for
    // now

    DECLARE_SCALAR_PARAM(T, k);
    DECLARE_SCALAR_PARAM(size_t, vlen);
};

typedef multiply_const<std::int16_t> multiply_const_ss;
typedef multiply_const<std::int32_t> multiply_const_ii;
typedef multiply_const<float> multiply_const_ff;
typedef multiply_const<gr_complex> multiply_const_cc;

} // namespace blocks
} // namespace gr
