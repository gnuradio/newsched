// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright 2020

#ifndef INCLUDED_TAG_HPP
#define INCLUDED_TAG_HPP

#include <cstdint>
#include <string>
#include <vector>

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
    tag_t(uint64_t offset, std::string& key, std::string& value, std::string& srcid)
    {
        offset = offset;
        key = key;
        value = value;
        srcid = srcid;
    }
    uint64_t offset;
    std::string key;
    // .... value  -- do without pmts for now
    std::string value;
    std::string srcid;
};
} // namespace gr

#endif