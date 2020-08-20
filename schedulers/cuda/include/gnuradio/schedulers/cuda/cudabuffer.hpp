#include <string.h>
#include <algorithm>
#include <cstdint>
#include <memory>
#include <mutex>
#include <vector>

#include <gnuradio/buffer.hpp>

namespace gr {
enum class cuda_buffer_type { D2D, H2D, D2H };

class cuda_buffer : public buffer
{
private:
    std::vector<uint8_t> _host_buffer;
    uint8_t* _device_buffer;
    unsigned int _read_index;
    unsigned int _write_index;
    unsigned int _num_items;
    unsigned int _item_size;
    unsigned int _buf_size;
    cuda_buffer_type _type;

    std::mutex _buf_mutex; // use raw mutex for now - FIXME - change to return mutex and
                           // used scoped lock outside on the caller
    cuda_buffer_type _buffer_type;

public:
    typedef std::shared_ptr<cuda_buffer> sptr;
    cuda_buffer(){};
    cuda_buffer(size_t num_items,
                size_t item_size,
                cuda_buffer_type type = cuda_buffer_type::D2D);
    ~cuda_buffer();

    static sptr make(size_t num_items, size_t item_size, cuda_buffer_type type = cuda_buffer_type::D2D);
    int size();
    int capacity();

    void* read_ptr();
    void* write_ptr();

    virtual bool read_info(buffer_info_t& info);

    virtual bool write_info(buffer_info_t& info);
    virtual void cancel();

    // virtual std::vector<tag_t> get_tags(unsigned int num_items);
    // virtual void add_tags(uint64_t offset, std::vector<tag_t>& tags);

    virtual void post_read(int num_items);
    virtual void post_write(int num_items);
    virtual void copy_items(std::shared_ptr<buffer> from, int nitems);
};

} // namespace gr