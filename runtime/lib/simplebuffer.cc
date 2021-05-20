#include <gnuradio/simplebuffer.hh>

namespace gr
{
    std::shared_ptr<buffer_reader> simplebuffer::add_reader()
    {
        std::shared_ptr<simplebuffer_reader> r(
            new simplebuffer_reader(shared_from_this(), _write_index));
        _readers.push_back(r.get());
        return r;
    }
}