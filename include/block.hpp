
// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright 2020

#include <cstdint>
#include <string>
#include <vector>

#include "io_signature.hpp"
#include "block_work_io.hpp"

namespace gr {

enum class work_return_code_t {
  WORK_DONE = -1,
  WORK_INSUFFICIENT_OUTPUT = 0,
  WORK_OK
};

class block {

protected:
  std::string d_name;
  block(const std::string &name, gr::io_signature &input_signature,
        gr::io_signature &output_signature);

  gr::io_signature d_input_signature, d_input_signature_capability;
  gr::io_signature d_output_signature, d_output_signature_capability;


  std::string d_name;

  virtual work_return_code_t work(block_work_io& work_input,
                                  block_work_io& work_output);

  virtual bool start();
  virtual bool stop();

  void set_relative_rate(double relative_rate);
  void set_relative_rate(unsigned int numerator, unsigned int denominator);

  uint64_t nitems_read(unsigned int which_input);
  uint64_t nitems_written(unsigned int which_output);

//   tag_propagation_policy_t tag_propagation_policy();
//   void set_tag_propagation_policy(tag_propagation_policy_t p);

public:
  ~block() {}

  static gr::io_signature& input_signature_capability();
  static gr::io_signature& output_signature_capability();
};

} // namespace gr