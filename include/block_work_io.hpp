
// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright 2020

#ifndef INCLUDED_BLOCK_WORK_IO_HPP
#define INCLUDED_BLOCK_WORK_IO_HPP

#include "tag.hpp"
#include <vector>


// Might also be a class if methods are needed
struct block_work_io {
  int n_items;
  uint64_t n_items_total;  // Name TBD. Replacement for _read and _written because I/O
  const void *items;
  std::vector<tag_t> &tags;
};

} // namespace gr

#endif
