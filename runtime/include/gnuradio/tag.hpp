#pragma once

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
    uint64_t offset;
    std::string key;
    // .... value  -- do without pmts for now - string is just a placeholder
    std::string value;
    std::string srcid;
    tag_t(uint64_t offset, std::string key, std::string value, std::string srcid = "")
        : offset(offset), key(key), value(value), srcid(srcid)
    {
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

