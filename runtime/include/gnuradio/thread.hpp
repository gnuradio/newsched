#include <sys/prctl.h>
#include <pthread.h>
#include <string>
#include <boost/format.hpp>

namespace gr {
namespace thread {
void set_thread_name(pthread_t thread, std::string name)
{
    if (thread != pthread_self()) // Naming another thread is not supported
        return;

    if (name.empty())
        name = boost::str(boost::format("thread %llu") % ((unsigned long long)thread));

    const int max_len = 16; // Maximum accepted by PR_SET_NAME

    if ((int)name.size() > max_len) // Shorten the name if necessary by taking as many
                                    // characters from the front
    {                               // so that the unique_id can still fit on the end
        int i = name.size() - 1;
        for (; i >= 0; --i) {
            std::string s = name.substr(i, 1);
            int n = atoi(s.c_str());
            if ((n == 0) && (s != "0"))
                break;
        }

        name = name.substr(0, std::max(0, max_len - ((int)name.size() - (i + 1)))) +
               name.substr(i + 1);
    }

    prctl(PR_SET_NAME, name.c_str(), 0, 0, 0);
}


} // namespace thread
} // namespace gr