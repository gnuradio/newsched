#pragma once

#include <gnuradio/streamops/stream_to_streams.hh>

namespace gr {
namespace streamops {

class stream_to_streams_cpu : public stream_to_streams
{
public:
    stream_to_streams_cpu(const block_args& args);
    
    virtual work_return_code_t work(std::vector<block_work_input>& work_input,
                                    std::vector<block_work_output>& work_output) override;

private:
    size_t d_itemsize;

};


} // namespace streamops
} // namespace gr
