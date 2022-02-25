GNU Radio ships with a default runtime (gr/include/gnuradio/runtime.h)

By separating out the runtime as a modular concept, different systems can be targeted in terms of how compute resources are allocated to signal processing blocks

This folder should hold non-default runtimes, which don't even need to be written in C++