#include <gnuradio/simplebuffer.hh>

namespace gr
{
    std::shared_ptr<buffer_reader> simplebuffer::add_reader(std::shared_ptr<buffer_properties> buf_props, size_t itemsize)
    {
        std::shared_ptr<simplebuffer_reader> r(
            new simplebuffer_reader(shared_from_this(), buf_props, itemsize, _write_index));
        _readers.push_back(r.get());
        return r;
    }
}