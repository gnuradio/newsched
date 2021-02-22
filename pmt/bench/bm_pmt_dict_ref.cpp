#include <chrono>
#include <iostream>
#include <string>

#include <boost/program_options.hpp>
namespace po = boost::program_options;

#include <pmt/pmtf.hpp>
#include <pmt/pmtf_map.hpp>
#include <pmt/pmtf_scalar.hpp>

using namespace pmtf;

bool run_test(const int times, pmt_map<std::string>::sptr d, uint64_t index)
{
    std::stringbuf sb; // fake channel

    auto key = std::string("key"+std::to_string(index));

    bool valid = true;
    for (int i=0; i< times; i++)
    {
        auto ref = d->ref(key);

        // if (ref == nullptr)
        //    valid = false;

        if (std::static_pointer_cast<pmt_scalar<int32_t>>(ref)->value() != index)
        {
            valid = false;
        }
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
        "index",
        po::value<uint64_t>(&index)->default_value(0),
        "Index for lookup");


    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    if (vm.count("help")) {
        std::cout << desc << std::endl;
        return 0;
    }

    {
        // Create the dictionary
        std::map<std::string,pmt_sptr> starting_map;
        for (int k = 0; k < items; k++)
        {
            auto key = std::string("key" + std::to_string(k));
            auto value = pmt_scalar<int32_t>::make(k);

            starting_map[key] = value;
        }

        auto d = pmt_map<std::string>::make(starting_map);

        auto t1 = std::chrono::steady_clock::now();

        auto valid = run_test(samples, d, index);

        auto t2 = std::chrono::steady_clock::now();
        auto time =
            std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count() / 1e9;

        std::cout << "[PROFILE_TIME]" << time << "[PROFILE_TIME]" << std::endl;
        std::cout << "[PROFILE_VALID]" << valid << "[PROFILE_VALID]" << std::endl;
    }
}
