#include <chrono>
#include <iostream>
#include <thread>

#include <gnuradio/blocklib/blocks/multiply_const.hpp>
#include <gnuradio/blocklib/blocks/vector_sink.hpp>
#include <gnuradio/blocklib/blocks/vector_source.hpp>
#include <gnuradio/domain_adapter_shm.hpp>
#include <gnuradio/flowgraph.hpp>
#include <gnuradio/schedulers/simplestream/scheduler_simplestream.hpp>

using namespace gr;

int main(int argc, char* argv[])
{
    float k1 = 2.0;
    int nsamps = 100;
    std::vector<float> input_data(nsamps);
    std::vector<float> expected_output(nsamps);
    for (int i = 0; i < nsamps; i++) {
        input_data[i] = i;
        expected_output[i] = i * k1 ;
    }

    if (1) {
        std::vector<tag_t> tags{ tag_t(17,"key1","value1"), tag_t(39,"key2","value2"), tag_t(84,"key3","value3")};
        auto src = blocks::vector_source_f::make(input_data, false, 1, tags);
        auto mult = blocks::multiply_const_ff::make(k1);

        auto snk = blocks::vector_sink_f::make();

        flowgraph_sptr fg(new flowgraph());
        fg->connect(src, 0, mult, 0);
        fg->connect(mult, 0,  snk, 0);


        std::shared_ptr<schedulers::scheduler_simplestream> sched1(
            new schedulers::scheduler_simplestream("sched1"));

        fg->add_scheduler(sched1);

        

        fg->validate();

        auto t1 = std::chrono::steady_clock::now();
        fg->start();
        fg->wait();

        auto t2 = std::chrono::steady_clock::now();
        std::chrono::duration<double, std::milli> fp_ms = t2 - t1;
        std::cout << "Domain Adapter flowgraph took: " << fp_ms.count() << std::endl;

        auto vec = snk->data();
        std::cout << (vec == expected_output) << " " << vec.size() << " "
                  << expected_output.size() << std::endl;


        auto out_tags = snk->tags();
        std::cout << "tags_received: " << out_tags.size() << " same: " << ((tags == out_tags) ? "true" : "false") << std::endl;


        std::cout << std::endl;
    }
  
}
