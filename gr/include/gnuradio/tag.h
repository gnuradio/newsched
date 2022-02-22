#pragma once

#include <pmtf/wrap.hpp>
#include <string>

namespace gr {

enum class tag_propagation_policy_t {
    TPP_DONT = 0,       /*!< Scheduler doesn't propagate tags from in- to output. The
                           block itself is free to insert tags as it wants. */
    TPP_ALL_TO_ALL = 1, /*!< Propagate tags from all in- to all outputs. The
                           scheduler takes care of that. */
    TPP_ONE_TO_ONE = 2, /*!< Propagate tags from n. input to n. output. Requires
                           same number of in- and outputs */
    TPP_CUSTOM = 3      /*!< Like TPP_DONT, but signals the block it should implement
                           application-specific forwarding behaviour. */
};

class tag_t
{
public:
    uint64_t offset = 0;
    pmtf::pmt key = nullptr;
    pmtf::pmt value = nullptr;
    pmtf::pmt srcid = nullptr;
    bool modified = false;
    tag_t() {}
    tag_t(uint64_t offset, pmtf::pmt key, pmtf::pmt value, pmtf::pmt srcid = nullptr)
        : offset(offset), key(key), value(value), srcid(srcid)
    {
    }

    /*!
     * Comparison function to test which tag, \p x or \p y, came
     * first in time
     */
    static inline bool offset_compare(const tag_t& x, const tag_t& y)
    {
        return x.offset < y.offset;
    }

    bool operator==(const tag_t& rhs) const
    {
        return (rhs.key == key && rhs.value == value && rhs.srcid == srcid);
    }
    bool operator!=(const tag_t& rhs) const
    {
        return (rhs.key != key || rhs.value != value || rhs.srcid == srcid);
    }
};

} // namespace gr
