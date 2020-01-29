
// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright 2020

#ifndef INCLUDED_BLOCK_WORK_IO_HPP
#define INCLUDED_BLOCK_WORK_IO_HPP

#include "tag.hpp"
#include <vector>


// Might also be a class if methods are needed
struct block_work_input {
  const int n_items;
  const uint64_t n_items_read;  // Name TBD. Replacement for _read and _written because I/O
  const void *items; // cannot be const for output items
  std::vector<tag_t> &tags;
  int& n_consumed; // output the number of items that were consumed on the work() call
};

struct block_work_output {
  const int n_items;
  const uint64_t n_items_written;  // Name TBD. Replacement for _read and _written because I/O
  void *items; // cannot be const for output items
  std::vector<tag_t> &tags;
  int& n_produced; // output the number of items that were produced in the work() call
};

} // namespace gr

#endif
