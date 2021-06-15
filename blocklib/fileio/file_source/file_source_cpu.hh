#pragma once

#include <gnuradio/fileio/file_source.hh>

namespace gr {
namespace fileio {

class file_source_cpu : public file_source
{
public:
    file_source_cpu(block_args args) : file_source(args), d_itemsize(args.itemsize) {}
    virtual work_return_code_t work(std::vector<block_work_input>& work_input,
                                    std::vector<block_work_output>& work_output) override;

protected:
    size_t d_itemsize;
};

} // namespace fileio
} // namespace gr