# The Flowgraph

Reference [commit](https://github.com/gnuradio/newsched/commit/8bb970e59fd30baded00b6bf96a7f5fc23a663ad)

The flowgraph is the highest level object, and what could be considered "launching an instance" of newsched.

Flowgraph derives from _graph_, as it is used to connect _blocks_ together, and start and stop execution.  It is also given a scheduler object(s) to handle execution of the flowgraph

The constructor is empty, but the main interface methods are:

```cpp
void set_scheduler(scheduler_sptr sched);
void validate();
void start();
void stop();
void wait();
```

(more methods for setting multiple schedulers will be shown later)

### set_scheduler
Simply stores the scheduler pointer for use by the execution commands

### validate
Kicks off the initialization of the scheduler(s).  Used to indicate that flowgraph configuration is complete

### start/stop/wait
Passes through start/stop/wait to the schedulers and the Flowgraph Monitor

---

As you can see, there is not much involved in the flowgraph classes, and all of the functionality comes from its parent classes (e.g. graph/node), and the schedulers.

Since `flowgraph` derives from `graph`, the `connect()` API is used to create edges and build a graph.  This graph is then flattened and handed off to the scheduler.
