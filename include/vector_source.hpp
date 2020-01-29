#ifndef INCLUDED_VECTOR_SOURCE_HPP
#define INCLUDED_VECTOR_SOURCE_HPP

#include "sync_block.hpp"

namespace gr {
namespace blocks {
template <class T> class vector_source : virtual public sync_block {
private:
  std::vector<T> d_data;
  bool d_repeat;
  unsigned int d_offset;
  unsigned int d_vlen;
  bool d_settags;
  std::vector<tag_t> d_tags;

public:
  vector_source(const std::vector<T> &data, bool repeat, unsigned int vlen,
                const std::vector<tag_t> &tags);
  ~vector_source();

  virtual work_return_code_t work(block_work_io &work_input,
                                  block_work_io &work_output);

  void rewind();
  void set_data(const std::vector<T> &data,
                const std::vector<tag_t> &tags = std::vector<tag_t>());
  void set_repeat(bool repeat);
};

typedef vector_source<std::uint8_t> vector_source_b;
typedef vector_source<std::int16_t> vector_source_s;
typedef vector_source<std::int32_t> vector_source_i;
typedef vector_source<float> vector_source_f;
typedef vector_source<gr_complex> vector_source_c;

} // namespace blocks
} // namespace gr
#endif