#pragma once

namespace gr {

enum class scheduler_action_t { DONE, NOTIFY_OUTPUT, NOTIFY_INPUT, NOTIFY_ALL, EXIT };


enum class scheduler_message_t {
    SCHEDULER_ACTION,
    ASYNC_MESSAGE,
    PARAMETER_QUERY,
    PARAMETER_CHANGE,
    CALLBACK
};

class scheduler_message
{
public:
    scheduler_message(scheduler_message_t type) : _type(type) {}
    scheduler_message_t type() { return _type; }
    int64_t blkid() { return _blkid; }

private:
    scheduler_message_t _type;
    int64_t _blkid;

    // scheduler message can be of type:
    //   parameter_query
    //   parameter_change
    //   callback
    //   scheduler_action
    //   async message (pmt)
};

typedef std::shared_ptr<scheduler_message> scheduler_message_sptr;

class scheduler_action : public scheduler_message
{
public:
    scheduler_action(scheduler_action_t action)
        : scheduler_message(scheduler_message_t::SCHEDULER_ACTION), _action(action)
    {
    }
    scheduler_action_t action() { return _action; }

private:
    scheduler_action_t _action;
};

typedef std::shared_ptr<scheduler_action> scheduler_action_sptr;

} // namespace gr