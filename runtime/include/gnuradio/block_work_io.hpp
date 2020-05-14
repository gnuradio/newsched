// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright 2020

#ifndef INCLUDED_BLOCK_WORK_IO_HPP
#define INCLUDED_BLOCK_WORK_IO_HPP

#include "tag.hpp"
#include <vector>
#include <cstdint>
#include <functional>

namespace gr
{

// Might also be a class if methods are needed
struct block_work_input {
  int n_items;
  uint64_t n_items_read;  // Name TBD. Replacement for _read and _written because I/O
  void *items; // cannot be const for output items
  std::vector<tag_t> &tags;
  int n_consumed; // output the number of items that were consumed on the work() call

  block_work_input(  int _n_items,uint64_t _n_items_read, void *_items, std::vector<tag_t> &_tags) 
  : n_items(_n_items), n_items_read(_n_items_read), items(_items), tags(_tags), n_consumed(-1) {}
};

struct block_work_output {
  int n_items;
  uint64_t n_items_written;  // Name TBD. Replacement for _read and _written because I/O
  void *items; // cannot be const for output items
  std::vector<tag_t> &tags;
  int n_produced; // output the number of items that were consumed on the work() call

  block_work_output(  int _n_items,uint64_t _n_items_written, void *_items, std::vector<tag_t> &_tags) 
  : n_items(_n_items), n_items_written(_n_items_written), items(_items), tags(_tags), n_produced(-1) {}
};

} // namespace gr

#endif
