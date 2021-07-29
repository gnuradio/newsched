#pragma once

#include <gnuradio/dtv/atsc_fpll.hh>

#include <gnuradio/math/nco.hh>
#include <gnuradio/filter/single_pole_iir.hh>

namespace gr {
namespace dtv {

class atsc_fpll_cpu : public atsc_fpll
{
public:
    atsc_fpll_cpu(const block_args& args);
    virtual work_return_code_t work(std::vector<block_work_input>& work_input,
                                    std::vector<block_work_output>& work_output) override;

private:
    gr::nco<float, float> d_nco;
    gr::filter::kernel::single_pole_iir<gr_complex, gr_complex, float> d_afc;
};

} // namespace dtv
} // namespace gr