#include <chrono>
#include <iostream>
#include <thread>

#include <gnuradio/blocklib/blocks/multiply_const_blk.hpp>
#include <gnuradio/blocklib/blocks/throttle.hpp>
#include <gnuradio/blocklib/blocks/vector_sink.hpp>
#include <gnuradio/blocklib/blocks/vector_source.hpp>
#include <gnuradio/flowgraph.hpp>
#include <gnuradio/schedulers/simplestream/scheduler_simplestream.hpp>

using namespace gr;

int main(int argc, char* argv[])
{
    std::shared_ptr<blocks::multiply_const_ff> mult(
        new blocks::multiply_const_ff(100.0)); // create a block that multiplies by 17
    std::shared_ptr<blocks::vector_source_f> src(new blocks::vector_source_f(
        std::vector<float>({ 1.0, 2.0, 3.0, 4.0, 5.0 }), true));
    std::shared_ptr<blocks::throttle> throttle(new blocks::throttle(sizeof(float), 100));
    std::shared_ptr<blocks::vector_sink_f> snk(new blocks::vector_sink_f());
    // blocks::vector_sink_f snk();

    // mult->on_parameter_change(std::vector<param_action_base>{
    // param_action<float>(blocks::multiply_const_ff::params::id_k, 12.0, 0) });

    flowgraph_sptr fg(new flowgraph());
    fg->connect(src->base(), 0, mult->base(), 0);
    fg->connect(mult->base(), 0, throttle->base(), 0);
    fg->connect(throttle->base(), 0, snk->base(), 0);
    // fg->connect(mult->base(), 0, snk->base(), 0);


    std::shared_ptr<schedulers::scheduler_simplestream> sched(
        new schedulers::scheduler_simplestream());
    fg->set_scheduler(sched->base());

    fg->validate();
    fg->start();

    auto start = std::chrono::steady_clock::now();

    float k = 1.0;
    while (true) {
        std::cout << "query k is " << mult->k() << std::endl;

        mult->set_k(k);

        mult->do_a_bunch_of_things(1,2.0,std::vector<gr_complex>{gr_complex(4.0,5.0)});

        if (std::chrono::steady_clock::now() - start > std::chrono::seconds(200))
            break;

        std::this_thread::sleep_for(std::chrono::milliseconds(2000));

        k += 1.0;

        for (const auto& d : snk->data())
            std::cout << d << ' ';
        std::cout << std::endl;
    }

    for (const auto& d : snk->data())
        std::cout << d << ' ';

    fg->stop();
    fg->wait();

    // DOMAIN??


    // sched.start();
    // sched.wait();

    for (const auto& d : snk->data())
        std::cout << d << ' ';
    std::cout << std::endl;

    // now look at the data
}