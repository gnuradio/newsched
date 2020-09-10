#pragma once

#include <condition_variable>
#include <mutex>
#include <deque>
#include <atomic>
#include <iostream>

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
        // std::cout << "**pop" << std::endl;
        std::unique_lock<std::mutex> l(_mutex);
        _cond.wait(l, [this] { return !_queue.empty(); }); // TODO - replace with a waitfor

        // std::cout << "qsz: " << _queue.size() << std::endl;
        msg = _queue.front();
        _queue.pop_front();
        return true;
    }
    void clear()
    {
        std::unique_lock<std::mutex> l(_mutex);
        _queue.clear();
        // l.unlock();
        // _cond.notify_all();
    }

private:
    std::deque<T> _queue;
    std::mutex _mutex;
    std::condition_variable _cond;
    std::atomic<bool> _ready;
};
} // namespace gr