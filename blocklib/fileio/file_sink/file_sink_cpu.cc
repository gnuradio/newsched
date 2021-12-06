#include "file_sink_cpu.hh"
#include "file_sink_cpu_gen.hh"

namespace gr {
namespace fileio {

file_sink_cpu::file_sink_cpu(const block_args& args)
    : sync_block("file_sink"),
      file_sink_base(args.filename, true, args.append),
      file_sink(args),
      d_itemsize(args.itemsize)
      
{
}

file_sink_cpu::~file_sink_cpu() {}

work_return_code_t file_sink_cpu::work(std::vector<block_work_input_sptr>& work_input,
                                       std::vector<block_work_output_sptr>& work_output)
{
    auto inbuf = work_input[0]->items<uint8_t>();
    auto noutput_items = work_input[0]->n_items;

    int nwritten = 0;

    do_update(); // update d_fp is reqd

    if (!d_fp) {
        work_input[0]->n_consumed = noutput_items; // drop output on the floor
        return work_return_code_t::WORK_OK;
    }


    while (nwritten < noutput_items) {
        const int count = fwrite(inbuf, d_itemsize, noutput_items - nwritten, d_fp);
        if (count == 0) {
            if (ferror(d_fp)) {
                std::stringstream s;
                s << "file_sink write failed with error " << fileno(d_fp) << std::endl;
                throw std::runtime_error(s.str());
            } else { // is EOF
                break;
            }
        }
        nwritten += count;
        inbuf += count * d_itemsize;
    }

    if (d_unbuffered)
        fflush(d_fp);

    work_input[0]->n_consumed = nwritten;
    return work_return_code_t::WORK_OK;
}

bool file_sink_cpu::stop()
{
    do_update();
    fflush(d_fp);
    return true;
}

} // namespace fileio
} // namespace gr