/* -*- c++ -*- */
/*
 * Copyright 2006,2007,2013 Free Software Foundation, Inc.
 *
 * This file is part of GNU Radio
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 */

#ifndef INCLUDED_GR_RUNTIME_FLOWGRAPH_H
#define INCLUDED_GR_RUNTIME_FLOWGRAPH_H

#include "api.h"
#include <iostream>
#include <memory>

#include <gnuradio/graph.hpp>
#include <gnuradio/scheduler.hpp>
namespace gr {

class flowgraph : public graph
{
private:
    scheduler_sptr d_sched;
    flat_graph_sptr d_flat_graph;
public:
    flowgraph() {};
    typedef std::shared_ptr<flowgraph> sptr;
    virtual ~flowgraph() {};
    void set_scheduler(scheduler_sptr sched)
    {
        d_sched = sched;
    }
    void validate()
    {   
        d_flat_graph = flat_graph::make_flat(base());
        d_sched->initialize(d_flat_graph);
    }
    void start() 
    {
        d_sched->start();
    }
    void stop() 
    {
        d_sched->stop();
    };
    void wait() 
    {
        d_sched->wait();
    };
};

typedef flowgraph::sptr flowgraph_sptr;
} // namespace gr

#endif