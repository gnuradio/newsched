# The Flowgraph

The flowgraph is just a `graph` (blocks and edges), but with some additional methods so that it can be used as the highest level of execution

Flowgraph derives from _graph_, as it is used to connect _blocks_ together, and start and stop execution.  It is also given a scheduler object(s) to handle execution of the flowgraph

The constructor only consists of a `name`, but the main interface methods are (as they wrap `runtime`):

```cpp
void start();
void stop();
void wait();
```

### check_connections

Static method used ensure that a graph is connected correctly.  This method also sets itemsizes of ports that are previously set to 0 by inferring the size based on connected ports.

### make_flat

Used to return the flattened representation of the graph

### start/stop/wait
If using the default runtime, passes through start/stop/wait to the runtime object (and subsequently the runtime monitor)

---

As you can see, there is not much involved in the flowgraph classes, and all of the functionality comes from its parent classes (e.g. graph/node), and the schedulers.

Since `flowgraph` derives from `graph`, the `connect()` API is used to create edges and build a graph.  This graph is then flattened and handed off to the scheduler.
