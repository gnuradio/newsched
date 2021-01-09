#pragma once

#include <condition_variable>
#include <atomic>
#include <deque>
#include <iostream>
#include <mutex>

#include <gnuradio/scheduler_message.hpp>

namespace gr {

template <typename T>
class concurrent_queue
{
public:
    bool push(const T& msg)
    {
        // std::cout << "**push" << std::endl;
        std::unique_lock<std::mutex> l(_mutex);
        _queue.push_back(msg);
        l.unlock();
        _cond.notify_all();

        return true;
    }
    bool pop(T& msg)
    {
        std::unique_lock<std::mutex> l(_mutex);

        if (_num_pop < 5) { // use the condition variable at the beginning
            _cond.wait(
                l, [this] { return !_queue.empty(); }); // TODO - replace with a waitfor
            msg = _queue.front();
            _queue.pop_front();
            _num_pop++;
            return true;
        }

        if (_queue.size() > 0) {
            msg = _queue.front();
            _queue.pop_front();
            return true;
        } else {
            return false;
        }
    }
    void clear()
    {
        std::unique_lock<std::mutex> l(_mutex);
        _queue.clear();
        // l.unlock();
        // _cond.notify_all();
    }

    size_t size()
    {
        std::unique_lock<std::mutex> l(_mutex);
        return _queue.size();
    }

private:
    std::deque<T> _queue;
    std::mutex _mutex;
    std::condition_variable _cond;
    std::atomic<bool> _ready;

    int _num_pop = 0;
};
} // namespace gr
