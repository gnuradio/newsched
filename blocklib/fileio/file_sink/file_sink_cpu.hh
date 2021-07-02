#pragma once

#include <gnuradio/fileio/file_sink.hh>
#include <gnuradio/fileio/file_sink_base.hh>

namespace gr {
namespace fileio {

class file_sink_cpu : public file_sink, virtual public file_sink_base
{
public:
    file_sink_cpu(const block_args& args);
    ~file_sink_cpu() override;

    bool stop() override;

    virtual work_return_code_t work(std::vector<block_work_input>& work_input,
                                    std::vector<block_work_output>& work_output) override;

    virtual void set_unbuffered(bool unbuffered)
    {
        file_sink_base::set_unbuffered(unbuffered);
    }


private:
    size_t d_itemsize;
};

} // namespace fileio
} // namespace gr