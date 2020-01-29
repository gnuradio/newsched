
// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright 2020

#ifndef INCLUDED_BLOCK_WORK_IO_HPP
#define INCLUDED_BLOCK_WORK_IO_HPP

#include "tag.hpp"
#include <vector>

namespace gr {
struct block_work_io {
  std::vector<int> &n_items;
  std::vector<const void *> &items;
  std::vector<tag_t> &tags;
};


} // namespace gr

#endif