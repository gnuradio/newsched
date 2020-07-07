#include <chrono>
#include <iostream>
#include <thread>

#include <gnuradio/blocklib/blocks/dummy.hpp>
#include <gnuradio/blocklib/blocks/throttle.hpp>
#include <gnuradio/blocklib/blocks/vector_sink.hpp>
#include <gnuradio/blocklib/blocks/vector_source.hpp>
#include <gnuradio/flowgraph.hpp>
#include <gnuradio/schedulers/simplestream/scheduler_simplestream.hpp>

using namespace gr;

int main(int argc, char* argv[])
{
    auto src = blocks::vector_source_f::make(
        std::vector<float>{ 1.0, 2.0, 3.0, 4.0, 5.0 }, false);
    auto snk1 = blocks::vector_sink_f::make();
    auto snk2 = blocks::vector_sink_f::make();


    flowgraph_sptr fg(new flowgraph());
    fg->connect(src->base(), 0, snk1->base(), 0);
    fg->connect(src->base(), 0, snk2->base(), 0);

    std::shared_ptr<schedulers::scheduler_simplestream> sched(
        new schedulers::scheduler_simplestream());
    fg->set_scheduler(sched->base());

    fg->validate();

    fg->start();
    fg->wait();

    for (const auto& d : snk1->data())
        std::cout << d << ' ';
    std::cout << std::endl;

    for (const auto& d : snk2->data())
        std::cout << d << ' ';
    std::cout << std::endl;

}