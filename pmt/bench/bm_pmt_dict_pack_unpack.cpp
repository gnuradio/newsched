#include <chrono>
#include <iostream>
#include <string>

#include <boost/program_options.hpp>
namespace po = boost::program_options;

#include <pmt/pmtf.hpp>
#include <pmt/pmtf_map.hpp>
#include <pmt/pmtf_scalar.hpp>

using namespace pmtf;

bool run_test(const int times, uint64_t nitems)
{

    bool valid = true;
    for (int i = 0; i < times; i++) {
        // Create the dictionary
        std::map<std::string, pmt_wrap> starting_map;

        #if 1
        for (uint64_t k = 0; k < nitems; k++) {
            auto key = std::string("key" + std::to_string(k));
            auto value = pmt_scalar(k);

            starting_map[key] = value;
        }
        auto d_in = pmt_map(starting_map);
        #else
        auto d_in = pmt_map<std::string>::make(starting_map);
        for (int k = 0; k < nitems; k++) {
            auto key = std::string("key" + std::to_string(k));
            auto value = pmt_scalar<int32_t>::make(k);

            d_in->set(key,value);
        }
        #endif

        #if 0
        auto d_out = d_in->value();

        for (int k = 0; k < nitems; k++) {
            auto key = std::string("key" + std::to_string(k));
            auto value = pmt_scalar<int32_t>::make(k);

            if (std::static_pointer_cast<pmt_scalar<int32_t>>(d_out[key])->value() != k) {
                valid = false;
            }
        }
        #endif
    }
    return valid;
}

int main(int argc, char* argv[])
{
    uint64_t samples;
    uint64_t items;
    uint64_t index;

    po::options_description desc("Basic Test Flow Graph");
    desc.add_options()("help,h", "display help")(
        "samples",
        po::value<uint64_t>(&samples)->default_value(10000),
        "Number of times to perform lookup")(
        "items",
        po::value<uint64_t>(&items)->default_value(100),
        "Number of items in dict")(
        "index", po::value<uint64_t>(&index)->default_value(0), "Index for lookup");


    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    if (vm.count("help")) {
        std::cout << desc << std::endl;
        return 0;
    }

    {

        auto t1 = std::chrono::steady_clock::now();

        auto valid = run_test(samples, items);

        auto t2 = std::chrono::steady_clock::now();
        auto time =
            std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count() / 1e9;

        std::cout << "[PROFILE_TIME]" << time << "[PROFILE_TIME]" << std::endl;
        std::cout << "[PROFILE_VALID]" << valid << "[PROFILE_VALID]" << std::endl;
    }
}
