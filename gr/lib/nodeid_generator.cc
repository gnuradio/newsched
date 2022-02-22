#include <gnuradio/nodeid_generator.h>

namespace gr {

std::string nodeid_generator::get_unique_string_()
{
    auto generate_len = 6;
    std::string rand_str(generate_len, '\0');
    auto randchar = []() -> char {
        const char charset[] = "0123456789"
                               "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                               "abcdefghijklmnopqrstuvwxyz";
        const size_t max_index = (sizeof(charset) - 1);
        return charset[rand() % max_index];
    };
    while (true) {
        // std::random_device random_device;
        // std::mt19937 generator{random_device()};
        // std::uniform_int_distribution<> distribution('a', '9');

        // auto x = distribution(generator);

        for (auto& dis : rand_str)
            dis = randchar();

        // verify that it is not in the used_strings;
        if (std::find(_used_strings.begin(), _used_strings.end(), rand_str) ==
            _used_strings.end()) {
            _used_strings.push_back(rand_str);
            break;
        }
        else {
            continue;
        }
    }

    return rand_str;
}

} // namespace gr