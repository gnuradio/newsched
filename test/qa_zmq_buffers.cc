#include <gtest/gtest.h>

#include <chrono>
#include <iostream>
#include <thread>

#include <gnuradio/blocks/copy.hh>
#include <gnuradio/math/multiply_const.hh>
#include <gnuradio/blocks/vector_sink.hh>
#include <gnuradio/blocks/vector_source.hh>
#include <gnuradio/blocks/head.hh>
#include <gnuradio/flowgraph.hh>
#include <gnuradio/schedulers/nbt/scheduler_nbt.hh>
#include <gnuradio/buffer_cpu_vmcirc.hh>
#include <gnuradio/buffer_net_zmq.hh>
#include <gnuradio/runtime.hh>

using namespace gr;

TEST(SchedulerMTTest, ZMQBuffers)
{
    size_t nsamples = 100000;
    std::vector<float> input_data(nsamples);
    for (size_t i = 0; i < nsamples; i++) {
        input_data[i] = i;
    }
    auto src = blocks::vector_source_f::make({input_data, true});
    // auto copy1 = blocks::copy::make({sizeof(float)});
    auto copy2 = blocks::copy::make({sizeof(float)});
    auto hd = blocks::head::make({nsamples, sizeof(float)});
    auto snk1 = blocks::vector_sink_f::make({});

    flowgraph_sptr fg(new flowgraph());
    // fg->connect(src, 0, copy1, 0);
    fg->connect(src, 0, copy2, 0)
        ->set_custom_buffer(buffer_net_zmq_properties::make("127.0.0.1", 1234));
    fg->connect(copy2, 0, hd, 0);
    fg->connect(hd, 0, snk1, 0);

    auto rt = runtime::make();
    rt->initialize(fg);
    rt->start();
    rt->wait();

    EXPECT_EQ(snk1->data().size(), input_data.size());
    EXPECT_EQ(snk1->data(), input_data);

    auto data = snk1->data();
    for (size_t i=0; i<data.size(); i++)
    {
        if (input_data[i] != data[i])
        {
            std::cout << i << ": " << input_data[i] << " " << data[i] << std::endl;
        }
    }
}
