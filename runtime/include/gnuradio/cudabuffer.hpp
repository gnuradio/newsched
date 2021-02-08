#include <string.h>
#include <algorithm>
#include <cstdint>
#include <memory>
#include <mutex>
#include <vector>

#include <gnuradio/buffer.hpp>

namespace gr {
enum class cuda_buffer_type { D2D, H2D, D2H };


class cuda_buffer_properties : public buffer_properties
{
public:
    // typedef sptr std::shared_ptr<buffer_properties>;
    cuda_buffer_properties(cuda_buffer_type buffer_type_)
        : buffer_properties(), _buffer_type(buffer_type_)
    {
    }
    cuda_buffer_type buffer_type() { return _buffer_type; }
    static std::shared_ptr<buffer_properties> make(cuda_buffer_type buffer_type_)
    {
        return std::dynamic_pointer_cast<buffer_properties>(
            std::make_shared<cuda_buffer_properties>(buffer_type_));
    }

private:
    cuda_buffer_type _buffer_type;
};


class cuda_buffer : public buffer
{
private:
    uint8_t* _host_buffer;
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

    cudaStream_t stream;

public:
    typedef std::shared_ptr<cuda_buffer> sptr;
    cuda_buffer(){};
    cuda_buffer(size_t num_items,
                size_t item_size,
                cuda_buffer_type type = cuda_buffer_type::D2D);
    ~cuda_buffer();

    static buffer_sptr make(size_t num_items,
                            size_t item_size,
                            std::shared_ptr<buffer_properties> buffer_properties);
    int size();
    int capacity();

    void* read_ptr();
    void* write_ptr();

    virtual bool read_info(buffer_info_t& info);

    virtual bool write_info(buffer_info_t& info);

    // virtual std::vector<tag_t> get_tags(unsigned int num_items);
    // virtual void add_tags(uint64_t offset, std::vector<tag_t>& tags);

    virtual void post_read(int num_items);
    virtual void post_write(int num_items);
    virtual void copy_items(std::shared_ptr<buffer> from, int nitems);
};


} // namespace gr

#define CUDA_BUFFER_ARGS_H2D cuda_buffer::make, cuda_buffer_properties::make(cuda_buffer_type::H2D)
#define CUDA_BUFFER_ARGS_D2H cuda_buffer::make, cuda_buffer_properties::make(cuda_buffer_type::D2H)
#define CUDA_BUFFER_ARGS_D2D cuda_buffer::make, cuda_buffer_properties::make(cuda_buffer_type::D2D)
