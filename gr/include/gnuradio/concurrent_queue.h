#pragma once

#include <blockingconcurrentqueue.h>

namespace gr {

/**
 * @brief Blocking Multi-producer Single-consumer Queue class
 *
 * @tparam T Data type of items in queue
 */
template <typename T>
class concurrent_queue
{
public:
    bool push(const T& msg)
    {
        q.enqueue(msg);
        return true;
    }

    // Non-blocking
    bool try_pop(T& msg)
    {
        return q.try_dequeue(msg);
    }
    bool pop(T& msg)
    {
        q.wait_dequeue(msg);
        return true;
    }
    void clear()
    {
        T msg;
        bool done = false;
        while(!done)
           done = !q.try_dequeue(msg);
    }
    size_t size_approx()
    {
        return q.size_approx();
    }

private:
    moodycamel::BlockingConcurrentQueue<T> q;
};
} // namespace gr
