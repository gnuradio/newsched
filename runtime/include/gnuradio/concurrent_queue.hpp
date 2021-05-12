#pragma once


#include <condition_variable>
#include <deque>
#include <iostream>
#include <mutex>

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
        std::unique_lock<std::mutex> l(_mutex);
        _queue.push_back(msg);
        l.unlock();
        _cond.notify_all();

        return true;
    }

    // Non-blocking
    bool try_pop(T& msg)
    {
        std::unique_lock<std::mutex> l(_mutex);
        if (!_queue.empty()) {
            msg = _queue.front();
            _queue.pop_front();
            return true;
        }
        else
        {
            return false;
        }
    }
    bool pop(T& msg)
    {
        std::unique_lock<std::mutex> l(_mutex);
        _cond.wait(l,
                   [this] { return !_queue.empty(); }); // TODO - replace with a waitfor
        msg = _queue.front();
        _queue.pop_front();
        return true;
    }
    void clear()
    {
        std::unique_lock<std::mutex> l(_mutex);
        _queue.clear();
    }

private:
    std::deque<T> _queue;
    std::mutex _mutex;
    std::condition_variable _cond;
};
} // namespace gr
