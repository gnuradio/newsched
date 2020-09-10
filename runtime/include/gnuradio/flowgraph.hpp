/* -*- c++ -*- */
/*
 * Copyright 2006,2007,2013 Free Software Foundation, Inc.
 *
 * This file is part of GNU Radio
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 */

#pragma once

#include "api.h"
#include <iostream>
#include <memory>

#include <condition_variable>
#include <mutex>
#include <queue>

#include <gnuradio/block.hpp>
#include <gnuradio/domain.hpp>
#include <gnuradio/domain_adapter.hpp>
#include <gnuradio/flowgraph_monitor.hpp>
#include <gnuradio/graph.hpp>
#include <gnuradio/scheduler.hpp>

namespace gr {

// typedef std::tuple<scheduler_sptr, std::vector<block_sptr>> partition_conf;
// typedef std::vector<partition_conf> partition_conf_vec;

class flowgraph : public graph
{
private:
    std::vector<scheduler_sptr> d_schedulers;
    flat_graph_sptr d_flat_graph;
    std::vector<graph_sptr> d_subgraphs;
    std::vector<flat_graph_sptr> d_flat_subgraphs;

    scheduler_sync _sched_sync;
    bool _monitor_thread_stopped = false;
    flowgraph_monitor_sptr d_fgmon;

public:
    flowgraph() { set_alias("flowgraph"); };
    typedef std::shared_ptr<flowgraph> sptr;
    virtual ~flowgraph() { _monitor_thread_stopped = true; };
    void set_scheduler(scheduler_sptr sched);
    void set_schedulers(std::vector<scheduler_sptr> sched);
    void add_scheduler(scheduler_sptr sched);
    void clear_schedulers();
    void partition(std::vector<domain_conf>& confs);
    void validate();
    void start();
    void stop();
    void wait();
};

typedef flowgraph::sptr flowgraph_sptr;
} // namespace gr
