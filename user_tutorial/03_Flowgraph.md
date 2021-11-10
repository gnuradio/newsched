# Creating and Running a Flowgraph

Flowgraphs define the blocks and connections that will perform the desired signal processing.  
The concepts here are the same as in GNU Radio

## Using GRC

The slightly modified version of GRC that is included with `newsched` should work the same as in GNU Radio.  
Differences will be documented as this integration is developed further

## Using Python

A flowgraph can be configured in a `python` script by instantiating blocks, connecting them together, and calling run on the flowgraph object.  For instance, let's consider a flowgraph with a `vector_source`, `vector_sink`, and `copy` block

```python
from newsched import gr, blocks

fg = gr.flowgraph("My Flowgraph")
src_data = [float(x) for x in range(0, 100)]

src = blocks.vector_source_f(src_data)
op = blocks.copy(gr.sizeof_float)
dst = blocks.vector_sink_f()

fg.connect([src, op, dst])
fg.run()

assert(src_data == dst.data())
```

Looks a lot like a GNU Radio flowgraph!!  That's because a lot of care has been made to make
the user experience similar between GNU Radio 3.x and newsched because the goal is for newsched
to prove out the long term vision for GNU Radio 4.x

We will next dig into the more advanced features of newsched, starting with how you might want to 
create your own module, blocks, and even custom scheduler