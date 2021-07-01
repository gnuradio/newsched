#include <gtest/gtest.h>

#include <chrono>
#include <iostream>
#include <thread>

#include <gnuradio/blocks/multiply_const.hh>
#include <gnuradio/blocks/vector_sink.hh>
#include <gnuradio/blocks/vector_source.hh>
#include <gnuradio/buffer.hh>
#include <gnuradio/buffer_cpu_simple.hh>
#include <gnuradio/flowgraph.hh>
#include <gnuradio/schedulers/mt/scheduler_mt.hh>

using namespace gr;


TEST(BlockDirectInterface, MultiplyConst)
{
    int nsamples = 1000000;
    std::vector<gr_complex> input_data(nsamples);
    std::vector<gr_complex> output_data(nsamples);
    std::vector<gr_complex> expected_data(nsamples);

    for (int i = 0; i < nsamples; i++) {
        input_data[i] = gr_complex(2 * i, 2 * i + 1);
        expected_data[i] = gr_complex(2 * i, 2 * i + 1);
    }

    auto src = blocks::vector_source_c::make({ input_data });
    auto snk = blocks::vector_sink_c::make();
    auto mult = blocks::multiply_const_cc::make_cpu(
        { gr_complex(1.0, 0.0) /* k */, 1 /* veclen */ });

    // Create vector of input/outputs
    std::vector<block_work_input> input_vec{ block_work_input(
        nsamples, sizeof(gr_complex), input_data.data()) };
    std::vector<block_work_output> output_vec{ block_work_output(
        nsamples, sizeof(gr_complex), output_data.data()) };

    // Call the work function
    mult->work(input_vec, output_vec);

    for (int i = 0; i < nsamples; i++)
        EXPECT_EQ(reinterpret_cast<gr_complex*>(output_vec[0].buffer->read_ptr(0))[i],
                  expected_data.data()[i]);
}
