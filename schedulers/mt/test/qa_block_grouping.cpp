#include <gtest/gtest.h>

#include <chrono>
#include <iostream>
#include <thread>

#include <gnuradio/blocklib/blocks/fanout.hpp>
#include <gnuradio/blocklib/blocks/multiply_const.hpp>
#include <gnuradio/blocklib/blocks/throttle.hpp>
#include <gnuradio/blocklib/blocks/vector_sink.hpp>
#include <gnuradio/blocklib/blocks/vector_source.hpp>
#include <gnuradio/domain_adapter_direct.hpp>
#include <gnuradio/flowgraph.hpp>
#include <gnuradio/schedulers/mt/scheduler_mt.hpp>
#include <gnuradio/vmcircbuf.hpp>

using namespace gr;

TEST(SchedulerBlockGrouping, BasicBlockGrouping)
{
    int nsamples = 1000000;
    std::vector<gr_complex> input_data(nsamples);
    std::vector<gr_complex> expected_data(nsamples);
    int buffer_type = 0;
    float k = 1.0;
    for (int i = 0; i < nsamples; i++) {
        input_data[i] = gr_complex(2 * i, 2 * i + 1);
        // expected_output[i] = gr_complex(k*2*i,k*2*i+1);
    }

    for (auto ngroups : { 2, 4, 8 }) {
        for (auto nblocks : { 2, 8, 16 }) {
            // for (auto nblocks : { 2, }) {
            int veclen = 1;
            auto src = blocks::vector_source_c::make(input_data);
            auto snk = blocks::vector_sink_c::make();
            std::vector<blocks::multiply_const_cc::sptr> mult_blks(nblocks * ngroups);

            for (int i = 0; i < nblocks * ngroups; i++) {
                mult_blks[i] = blocks::multiply_const_cc::make(k, veclen);
            }

            flowgraph_sptr fg(new flowgraph());

            auto sch = schedulers::scheduler_mt::make("mtsched");
            fg->connect(src, 0, mult_blks[0], 0)->set_custom_buffer(VMCIRC_BUFFER_ARGS);
            for (int n = 0; n < ngroups; n++) {
                std::vector<block_sptr> bg;
                for (int i = 0; i < nblocks; i++) {
                    int idx = n*nblocks +i;
                    if (idx > 0)
                    {
                        fg->connect(mult_blks[idx-1], 0, mult_blks[idx], 0)->set_custom_buffer(VMCIRC_BUFFER_ARGS);
                    }
                    bg.push_back(mult_blks[idx]);
                }
                sch->add_block_group(bg);
            }
            fg->connect(mult_blks[nblocks*ngroups-1], 0, snk, 0)->set_custom_buffer(VMCIRC_BUFFER_ARGS);
            
            fg->add_scheduler(sch);
            fg->validate();


            fg->start();
            fg->wait();

            EXPECT_EQ(snk->data(), input_data);
        }
    }
}
