#include <chrono>
#include <iostream>
#include <thread>

#include <gnuradio/blocklib/blocks/head.hpp>
#include <gnuradio/blocklib/blocks/multiply_const.hpp>
#include <gnuradio/blocklib/blocks/null_sink.hpp>
#include <gnuradio/blocklib/blocks/throttle.hpp>
#include <gnuradio/blocklib/blocks/vector_sink.hpp>
#include <gnuradio/blocklib/blocks/vector_source.hpp>
#include <gnuradio/domain_adapter_shm.hpp>
#include <gnuradio/flowgraph.hpp>
#include <gnuradio/logging.hpp>
#include <gnuradio/schedulers/st/scheduler_st.hpp>
#include <gnuradio/blocklib/cuda/multiply_const.hpp>
#include <gnuradio/blocklib/cuda/multiply_const2.hpp>

#include <gnuradio/cudabuffer.hpp>
#include <gnuradio/simplebuffer.hpp>

using namespace gr;

int main(int argc, char* argv[])
{
    auto logger = logging::get_logger("TEST_ST_CUDA", "debug");
    float k = 323.0;
    int nsamps = 5000000;
    std::vector<float> input_data(nsamps);
    std::vector<float> expected_output(nsamps);
    for (int i = 0; i < nsamps; i++) {
        input_data[i] = ((float)i);
        expected_output[i] = ((float)i * k);
    }

    std::cout << "*************************************************" << std::endl;
    if (0) {
        /// Test the CUDA version
        auto mult =
            cuda::multiply_const_ff::make(k); // create a block that multiplies by 17
        auto src = blocks::vector_source_f::make(input_data, false);
        auto snk = blocks::vector_sink_f::make();

        flowgraph_sptr fg(new flowgraph());
        fg->connect(src, 0, mult, 0);
        fg->connect(mult, 0, snk, 0);


        std::shared_ptr<schedulers::scheduler_st> sched(
            new schedulers::scheduler_st());
        fg->set_scheduler(sched);

        fg->validate();

        auto t1 = std::chrono::steady_clock::now();
        fg->start();
        fg->wait();

        if (snk->data() != expected_output) {
            std::cout << "Data mismatch" << std::endl;
        }

        auto t2 = std::chrono::steady_clock::now();
        std::chrono::duration<double, std::milli> fp_ms = t2 - t1;
        std::cout << "CUDA flowgraph took: " << fp_ms.count() << std::endl;
    }

    std::cout << "*************************************************" << std::endl;
    if (0) {
        /// Test the CUDA version
        auto mult =
            blocks::multiply_const_ff::make(k); // create a block that multiplies by 17
        auto src = blocks::vector_source_f::make(input_data, false);
        auto snk = blocks::vector_sink_f::make();

        flowgraph_sptr fg(new flowgraph());
        fg->connect(src, 0, mult, 0);
        fg->connect(mult, 0, snk, 0);

        std::shared_ptr<schedulers::scheduler_st> sched(
            new schedulers::scheduler_st());
        fg->set_scheduler(sched);

        fg->validate();

        auto t1 = std::chrono::steady_clock::now();
        fg->start();
        fg->wait();

        if (snk->data() != expected_output) {
            std::cout << "Data mismatch" << std::endl;
        }

        auto t2 = std::chrono::steady_clock::now();
        std::chrono::duration<double, std::milli> fp_ms = t2 - t1;
        auto int_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1);

        std::cout << "non-CUDA flowgraph took: " << fp_ms.count() << std::endl;
    }

    std::cout << "*************************************************" << std::endl;
    if (1) {
        float k1 = 2, k2 = 3, k3 = 4, k4 = 5;
        for (int i = 0; i < nsamps; i++) {
            expected_output[i] = ((float)i * k1 * k2 * k3 * k4);
        }

        /// Test the non-cuda version with the domain adapter
        // Add extra multiplies because we don't yet deal with orphan nodes
        auto mult1 = blocks::multiply_const_ff::make(k1);
        auto mult2 = blocks::multiply_const_ff::make(k2);
        auto mult3 = blocks::multiply_const_ff::make(k3);
        auto mult4 = blocks::multiply_const_ff::make(k4);
        auto src = blocks::vector_source_f::make(input_data, false);
        auto snk = blocks::vector_sink_f::make();

        flowgraph_sptr fg(new flowgraph());
        fg->connect(src, 0, mult1, 0);
        fg->connect(mult1, 0, mult2, 0);
        fg->connect(mult2, 0, mult3, 0);
        fg->connect(mult3, 0, mult4, 0);
        fg->connect(mult4, 0, snk, 0);


        std::shared_ptr<schedulers::scheduler_st> sched1(
            new schedulers::scheduler_st("sched1"));
        std::shared_ptr<schedulers::scheduler_st> sched2(
            new schedulers::scheduler_st("sched2"));
        fg->add_scheduler(sched1);
        fg->add_scheduler(sched2);

        auto da_conf = domain_adapter_shm_conf::make(buffer_preference_t::UPSTREAM);
        domain_conf_vec dconf{ domain_conf(sched1, { src, mult1, mult4, snk }, da_conf),
                               domain_conf(sched2, { mult2, mult3 }, da_conf) };
        fg->partition(dconf);

        auto t1 = std::chrono::steady_clock::now();
        fg->start();
        fg->wait();

        if (snk->data() != expected_output) {
            std::cout << "Data mismatch" << std::endl;
        } else {
            std::cout << "Data OK" << std::endl;
        }

        auto t2 = std::chrono::steady_clock::now();
        std::chrono::duration<double, std::milli> fp_ms = t2 - t1;
        std::cout << "non-CUDA with DA flowgraph took: " << fp_ms.count() << std::endl;
    }

    std::cout << "*************************************************" << std::endl;
    if (1) {
        float k1 = 2, k2 = 3, k3 = 4, k4 = 5;
        for (int i = 0; i < nsamps; i++) {
            expected_output[i] = ((float)i * k1 * k2 * k3 * k4);
        }

        /// Test the non-cuda version with the domain adapter
        // Add extra multiplies because we don't yet deal with orphan nodes
        auto mult1 = blocks::multiply_const_ff::make(k1);
        auto mult2 = cuda::multiply_const_ff::make(k2);
        auto mult3 = cuda::multiply_const_ff::make(k3);
        auto mult4 = blocks::multiply_const_ff::make(k4);
        auto src = blocks::vector_source_f::make(input_data, false);
        auto snk = blocks::vector_sink_f::make();

        flowgraph_sptr fg(new flowgraph());
        fg->connect(src, 0, mult1, 0);
        fg->connect(mult1, 0, mult2, 0);
        fg->connect(mult2, 0, mult3, 0);
        fg->connect(mult3, 0, mult4, 0);
        fg->connect(mult4, 0, snk, 0);


        std::shared_ptr<schedulers::scheduler_st> sched1(
            new schedulers::scheduler_st("sched1"));
        std::shared_ptr<schedulers::scheduler_st> sched2(
            new schedulers::scheduler_st("sched2"));
        fg->add_scheduler(sched1);
        fg->add_scheduler(sched2);

        auto da_conf = domain_adapter_shm_conf::make(buffer_preference_t::UPSTREAM);
        domain_conf_vec dconf{ domain_conf(sched1, { src, mult1, mult4, snk }, da_conf),
                               domain_conf(sched2, { mult2, mult3 }, da_conf) };
        fg->partition(dconf);

        auto t1 = std::chrono::steady_clock::now();
        fg->start();
        fg->wait();

        if (snk->data() != expected_output) {
            std::cout << "Data mismatch" << std::endl;
        } else {
            std::cout << "Data OK" << std::endl;
        }

        auto t2 = std::chrono::steady_clock::now();
        std::chrono::duration<double, std::milli> fp_ms = t2 - t1;
        std::cout << "CUDA with DA and extra copies flowgraph took: " << fp_ms.count() << std::endl;
    }

    std::cout << "*************************************************" << std::endl;
    {
        float k1 = 2, k2 = 3, k3 = 4, k4 = 5;
        for (int i = 0; i < nsamps; i++) {
            expected_output[i] = ((float)i * k1 * k2 * k3 * k4);
        }

        /// CUDA domain scheduler
        // Add extra multiplies because we don't yet deal with orphan nodes
        auto mult1 = blocks::multiply_const_ff::make(k1);
        auto mult2 = cuda::multiply_const2_ff::make(k2);
        auto mult3 = cuda::multiply_const2_ff::make(k3);
        auto mult4 = blocks::multiply_const_ff::make(k4);
        auto src = blocks::vector_source_f::make(input_data, false);
        auto snk = blocks::vector_sink_f::make();

        flowgraph_sptr fg(new flowgraph());
        fg->connect(src, 0, mult1, 0);
        fg->connect(mult1, 0, mult2, 0);
        fg->connect(mult2, 0, mult3, 0);
        fg->connect(mult3, 0, mult4, 0);
        fg->connect(mult4, 0, snk, 0);


        std::shared_ptr<schedulers::scheduler_st> sched1(
            new schedulers::scheduler_st("sched1"));
        std::shared_ptr<schedulers::scheduler_st> sched2(
            new schedulers::scheduler_st("sched2"));
        // std::shared_ptr<schedulers::scheduler_st> sched2(
        //     new schedulers::scheduler_st("sched2"));
        fg->add_scheduler(sched1);
        fg->add_scheduler(sched2);
        sched2->set_default_buffer_factory(cuda_buffer::make);

        auto da_conf_upstream =
            domain_adapter_shm_conf::make(buffer_preference_t::UPSTREAM);
        auto da_conf_downstream =
            domain_adapter_shm_conf::make(buffer_preference_t::DOWNSTREAM);

        // domain_conf_vec dconf{
        //     domain_conf(sched1, { src, mult1, mult4, snk }, da_conf_upstream),
        //     domain_conf(sched2,
        //                 { mult2, mult3 },
        //                 da_conf_downstream,
        //                 { std::make_tuple(edge(endpoint(mult1, 0), endpoint(mult2, 0)),
        //                                   da_conf_downstream) })
        // };

        domain_conf_vec dconf{
            domain_conf(sched1, { src, mult1, mult4, snk }, da_conf_upstream),
            domain_conf(sched2, { mult2, mult3 }, da_conf_downstream)
        };
        fg->partition(dconf);

        auto t1 = std::chrono::steady_clock::now();
        fg->start();
        fg->wait();

        if (snk->data() != expected_output) {
            std::cout << "Data mismatch" << std::endl;
        } else {
            std::cout << "Data OK" << std::endl;
        }

        auto t2 = std::chrono::steady_clock::now();
        std::chrono::duration<double, std::milli> fp_ms = t2 - t1;
        std::cout << "CUDA Domain Scheduler flowgraph took: " << fp_ms.count()
                  << std::endl;
    }
}
