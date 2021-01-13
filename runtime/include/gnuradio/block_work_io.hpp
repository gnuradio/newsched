#pragma once

#include <gnuradio/tag.hpp>
#include <cstdint>
#include <vector>

namespace gr {

/**
 * @brief Struct for passing all information needed for input data to block::work
 *
 */
struct block_work_input {
    int n_items;
    uint64_t n_items_read = 0; // Name TBD. Replacement for _read and _written because I/O
    void* items;           // cannot be const for output items
    std::vector<tag_t> tags;
    int n_consumed; // output the number of items that were consumed on the work() call

    /**
     * @brief Construct a new block work input object
     *
     * @param _n_items
     * @param _n_items_read
     * @param _items
     * @param _tags
     */
    block_work_input(int _n_items,
                     uint64_t _n_items_read,
                     void* _items,
                     std::vector<tag_t>& _tags)
        : n_items(_n_items),
          n_items_read(_n_items_read),
          items(_items),
          tags(_tags),
          n_consumed(-1)
    {
    }
};

/**
 * @brief Struct for passing all information needed for output data from block::work
 *
 */
struct block_work_output {
    int n_items;
    uint64_t n_items_written = 0; // Name TBD. Replacement for _read and _written because I/O
    void* items;              // cannot be const for output items
    std::vector<tag_t> tags;
    int n_produced; // output the number of items that were consumed on the work() call

    block_work_output(int _n_items,
                      uint64_t _n_items_written,
                      void* _items,
                      std::vector<tag_t>& _tags)
        : n_items(_n_items),
          n_items_written(_n_items_written),
          items(_items),
          tags(_tags),
          n_produced(-1)
    {
    }
};


/**
 * @brief Enum for return codes from calls to block::work
 *
 */
enum class work_return_code_t {
    WORK_ERROR = -100, /// error occurred in the work function
    WORK_INSUFFICIENT_OUTPUT_ITEMS =
        -3, /// work requires a larger output buffer to produce output
    WORK_INSUFFICIENT_INPUT_ITEMS =
        -2, /// work requires a larger input buffer to produce output
    WORK_DONE =
        -1, /// this block has completed its processing and the flowgraph should be done
    WORK_OK = 0, /// work call was successful and return values in i/o structs are valid
};

} // namespace gr
