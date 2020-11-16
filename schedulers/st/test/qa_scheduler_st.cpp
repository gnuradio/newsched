#include <gtest/gtest.h>

#include <chrono>
#include <iostream>
#include <thread>

#include <gnuradio/blocklib/blocks/fanout.hpp>
#include <gnuradio/blocklib/blocks/multiply_const.hpp>
#include <gnuradio/blocklib/blocks/throttle.hpp>
#include <gnuradio/blocklib/blocks/vector_sink.hpp>
#include <gnuradio/blocklib/blocks/vector_source.hpp>
#include <gnuradio/domain_adapter_shm.hpp>
#include <gnuradio/domain_adapter_direct.hpp>
#include <gnuradio/flowgraph.hpp>
#include <gnuradio/schedulers/st/scheduler_st.hpp>
#include <gnuradio/vmcirc_buffer.hpp>

using namespace gr;


TEST(SchedulerSTTest, TwoSinks)
{
    std::vector<float> input_data{ 1.0, 2.0, 3.0, 4.0, 5.0 };
    auto src = blocks::vector_source_f::make(input_data, false);
    auto snk1 = blocks::vector_sink_f::make();
    auto snk2 = blocks::vector_sink_f::make();


    flowgraph_sptr fg(new flowgraph());
    fg->connect(src, 0, snk1, 0);
    fg->connect(src, 0, snk2, 0);

    std::shared_ptr<schedulers::scheduler_st> sched(new schedulers::scheduler_st());
    fg->set_scheduler(sched);

    fg->validate();

    fg->start();
    fg->wait();

    EXPECT_EQ(snk1->data(), input_data);
    EXPECT_EQ(snk2->data(), input_data);
}


TEST(SchedulerSTTest, DomainAdapterBasic)
{
    std::vector<float> input_data{ 1.0, 2.0, 3.0, 4.0, 5.0 };

    std::vector<float> expected_data;
    for (auto d : input_data) {
        expected_data.push_back(100.0 * 200.0 * d);
    }

    auto src = blocks::vector_source_f::make(input_data, false);
    auto throttle = blocks::throttle::make(sizeof(float), 100);
    auto mult1 = blocks::multiply_const_ff::make(100.0);
    auto mult2 = blocks::multiply_const_ff::make(200.0);
    auto snk = blocks::vector_sink_f::make();

    flowgraph_sptr fg(new flowgraph());
    fg->connect(src, 0, mult1, 0);
    fg->connect(mult1, 0, mult2, 0);
    fg->connect(mult2, 0, snk, 0);

    std::shared_ptr<schedulers::scheduler_st> sched1(
        new schedulers::scheduler_st("sched1"));
    std::shared_ptr<schedulers::scheduler_st> sched2(
        new schedulers::scheduler_st("sched2"));

    fg->add_scheduler(sched1);
    fg->add_scheduler(sched2);

    auto da_conf = domain_adapter_shm_conf::make(buffer_preference_t::UPSTREAM);

    domain_conf_vec dconf{ domain_conf(sched1, { src, mult1 }, da_conf),
                           domain_conf(sched2, { mult2, snk }, da_conf) };

    fg->partition(dconf);

    fg->start();
    fg->wait();

    EXPECT_EQ(snk->data(), expected_data);
}

TEST(SchedulerSTTest, ParameterBasic)
{
    auto src = blocks::vector_source_f::make(
        std::vector<float>{ 1.0, 2.0, 3.0, 4.0, 5.0 }, true);
    auto throttle = blocks::throttle::make(sizeof(float), 32000);
    auto mult1 = blocks::multiply_const_ff::make(100.0);
    auto fanout = blocks::fanout::make(sizeof(float), 2);
    auto snk1 = blocks::vector_sink_f::make();
    auto snk2 = blocks::vector_sink_f::make();

    flowgraph_sptr fg(new flowgraph());
    fg->connect(src, 0, throttle, 0);
    fg->connect(throttle, 0, mult1, 0);
    fg->connect(mult1, 0, fanout, 0);
    fg->connect(fanout, 0, snk1, 0);
    fg->connect(fanout, 1, snk2, 0);

    std::shared_ptr<schedulers::scheduler_st> sched(
        new schedulers::scheduler_st("sched1", 100));
    fg->set_scheduler(sched);

    fg->validate();

    fg->start();

    auto start = std::chrono::steady_clock::now();

    float k = 1.0;

    while (true) {
        auto query_k = mult1->k();

        mult1->set_k(k);

        if (std::chrono::steady_clock::now() - start > std::chrono::seconds(1))
            break;

        std::this_thread::sleep_for(std::chrono::milliseconds(100));

        k += 1.0;
    }

    fg->stop();

    std::cout << "fg stopped" << std::endl;

    // now look at the data
    EXPECT_GT(snk1->data().size(), 5);
    EXPECT_GT(snk2->data().size(), 5);

    // TODO - check for query
    // TODO - set changes at specific sample numbers
}

TEST(SchedulerSTTest, BlockFanout)
{
    int nsamples = 1000000;
    std::vector<gr_complex> input_data(nsamples);
    std::vector<gr_complex> expected_data(nsamples);
    int buffer_type = 1;
    float k = 1.0;
    for (int i = 0; i < nsamples; i++) {
        input_data[i] = gr_complex(2 * i, 2 * i + 1);
        // expected_output[i] = gr_complex(k*2*i,k*2*i+1);
    }

    for (auto domain_adapt : { 0 }) {
        for (auto nblocks : { 2, 8, 16 }) {
            int veclen = 1;
            auto src = blocks::vector_source_c::make(input_data);
            std::vector<blocks::vector_sink_c::sptr> sink_blks(nblocks);
            std::vector<blocks::multiply_const_cc::sptr> mult_blks(nblocks);

            for (int i = 0; i < nblocks; i++) {
                mult_blks[i] = blocks::multiply_const_cc::make(k, veclen);
                sink_blks[i] = blocks::vector_sink_c::make();
            }
            flowgraph_sptr fg(new flowgraph());

            if (buffer_type == 0) {
                for (int i = 0; i < nblocks; i++) {
                    fg->connect(src, 0, mult_blks[i], 0);
                    fg->connect(mult_blks[i], 0, sink_blks[i], 0);
                }

            } else {
                for (int i = 0; i < nblocks; i++) {
                    fg->connect(src, 0, mult_blks[i], 0, VMCIRC_BUFFER_ARGS);
                    fg->connect(mult_blks[i], 0, sink_blks[i], 0, VMCIRC_BUFFER_ARGS);
                }
            }

            if (domain_adapt == 0) {
                std::cout << "no domain adapters" << std::endl;
                std::shared_ptr<schedulers::scheduler_st> sched1(
                    new schedulers::scheduler_st("sched1"));
                fg->add_scheduler(sched1);
                fg->validate();
            } else {
                auto da_conf =
                    domain_adapter_direct_conf::make(buffer_preference_t::DOWNSTREAM);
                std::vector<std::shared_ptr<schedulers::scheduler_st>> scheds(
                    nblocks * 2 + 1);
                for (int i = 0; i < 1 + 2 * nblocks; i++) {
                    scheds[i] = std::make_shared<schedulers::scheduler_st>(
                        "sched" + std::to_string(i + 1), 32768);
                    scheds[i]->set_default_buffer_factory(VMCIRC_BUFFER_ARGS);
                    fg->add_scheduler(scheds[i]);
                }

                domain_conf_vec dconf;
                for (int i = 0; i < nblocks; i++) {
                    dconf.push_back(
                        domain_conf(scheds[2 * i], { mult_blks[i] }, da_conf));
                    dconf.push_back(
                        domain_conf(scheds[2 * i + 1], { sink_blks[i] }, da_conf));
                }
                dconf.push_back(domain_conf(scheds[2 * nblocks], { src }, da_conf));

                fg->partition(dconf);
            }

            fg->start();
            fg->wait();

            for (int n = 0; n < nblocks; n++) {
                for (int i = 0; i < nsamples; i++) {
                    input_data[i] = gr_complex(2 * i, 2 * i + 1);
                    expected_data[i] = gr_complex(k*2*i,k*(2*i+1));
                }

                EXPECT_EQ(sink_blks[n]->data(), expected_data);
            }
        }
    }
}
