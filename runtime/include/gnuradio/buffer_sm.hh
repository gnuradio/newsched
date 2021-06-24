#pragma once

#include <string.h>
#include <algorithm>
#include <cstdint>
#include <memory>
#include <vector>

#include <gnuradio/buffer.hh>
// #include <gnuradio/logging.hh>

namespace gr {

typedef void* (*memcpy_func_t)(void* dest, const void* src, std::size_t count);
typedef void* (*memmove_func_t)(void* dest, const void* src, std::size_t count);

class buffer_sm_reader;
class buffer_sm : public buffer
{
private:
    std::vector<uint8_t> _buffer;

    // logger_sptr _logger;
    // logger_sptr _debug_logger;

public:
    typedef std::shared_ptr<buffer_sm> sptr;
    buffer_sm(size_t num_items,
              size_t item_size,
              std::shared_ptr<buffer_properties> buf_properties);

    static buffer_sptr
    make(size_t num_items,
         size_t item_size,
         std::shared_ptr<buffer_properties> buffer_properties = nullptr);

    void* read_ptr(size_t index);
    void* write_ptr();

    virtual void post_write(int num_items) override;

    virtual bool
    output_blocked_callback_logic(bool force = false,
                                  memmove_func_t memmove_func = std::memmove);

    virtual bool output_blocked_callback(bool force = false);
    virtual size_t space_available() override;

    virtual bool write_info(buffer_info_t& info) override;
    virtual std::shared_ptr<buffer_reader>
    add_reader(std::shared_ptr<buffer_properties> buf_props, size_t itemsize) override;

    bool adjust_buffer_data(memcpy_func_t memcpy_func, memmove_func_t memmove_func);
};

class buffer_sm_reader : public buffer_reader
{
private:
    // logger_sptr _logger;
    // logger_sptr _debug_logger;

protected:
    std::shared_ptr<buffer_sm> _buffer_sm;

public:
    buffer_sm_reader(std::shared_ptr<buffer_sm> buffer,
                     size_t itemsize,
                     std::shared_ptr<buffer_properties> buf_props = nullptr,
                     size_t read_index = 0);

    virtual void post_read(int num_items) override;

    virtual bool input_blocked_callback(size_t items_required);
    virtual size_t items_available() override;
};


class buffer_sm_properties : public buffer_properties
{
public:
    buffer_sm_properties();

    static std::shared_ptr<buffer_properties> make();
};


#define SM_BUFFER_ARGS buffer_sm_properties::make()

} // namespace gr
