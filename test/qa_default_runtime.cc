#include <gtest/gtest.h>

#include <chrono>
#include <iostream>
#include <thread>

#include <gnuradio/blocks/copy.hh>
#include <gnuradio/math/multiply_const.hh>
#include <gnuradio/blocks/vector_sink.hh>
#include <gnuradio/blocks/vector_source.hh>
#include <gnuradio/flowgraph.h>
#include <gnuradio/schedulers/nbt/scheduler_nbt.hh>
#include <gnuradio/buffer_cpu_vmcirc.h>
#include <gnuradio/runtime.h>

using namespace gr;

TEST(DefaultRuntimeTest, SeparateRuntime)
{
    int nsamples = 100000;
    std::vector<float> input_data(nsamples);
    for (int i = 0; i < nsamples; i++) {
        input_data[i] = i;
    }
    auto src = blocks::vector_source_f::make({input_data, false});
    auto blk = blocks::copy::make({});
    auto snk = blocks::vector_sink_f::make({});

    auto fg = flowgraph::make();
    fg->connect({src, blk, snk});

    auto rt = runtime::make();
    rt->initialize(fg);
    rt->start();
    rt->wait();

    EXPECT_EQ(snk->data().size(), input_data.size());
    EXPECT_EQ(snk->data(), input_data);

}
