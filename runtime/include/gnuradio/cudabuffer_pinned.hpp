#include <string.h>
#include <algorithm>
#include <cstdint>
#include <memory>
#include <mutex>
#include <vector>

#include <gnuradio/buffer.hpp>

namespace gr {

class cuda_buffer_pinned_properties : public buffer_properties
{
public:
    // typedef sptr std::shared_ptr<buffer_properties>;
    cuda_buffer_pinned_properties()
        : buffer_properties()
    {
    }
    static std::shared_ptr<buffer_properties> make()
    {
        return std::dynamic_pointer_cast<buffer_properties>(
            std::make_shared<cuda_buffer_pinned_properties>());
    }
};


class cuda_buffer_pinned : public buffer
{
private:
    uint8_t* _pinned_buffer;
    unsigned int _read_index;
    unsigned int _write_index;
    unsigned int _num_items;
    unsigned int _item_size;
    unsigned int _buf_size;

    std::mutex _buf_mutex; // use raw mutex for now - FIXME - change to return mutex and
                           // used scoped lock outside on the caller

public:
    typedef std::shared_ptr<cuda_buffer_pinned> sptr;
    cuda_buffer_pinned(){
        set_type("cuda_buffer_pinned");
    };
    cuda_buffer_pinned(size_t num_items,
                size_t item_size);
    ~cuda_buffer_pinned();

    static buffer_sptr make(size_t num_items,
                            size_t item_size,
                            std::shared_ptr<buffer_properties> buffer_properties);
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

#define CUDA_BUFFER_PINNED_ARGS cuda_buffer_pinned::make, cuda_buffer_pinned_properties::make()