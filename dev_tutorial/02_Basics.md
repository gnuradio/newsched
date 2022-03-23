# Basic Structure

## Licensing

Because newsched will be borrowing heavily from GNU radio, we stick with the GPLv3 license to ensure no license incompatibility issues

## Caveats

You will see in the implementation that many many things are constructed as `shared_ptr`s due to their simplicity.  This can be cleaned up.  There are other non-optimal implementations, as the current state is to establish the core concepts.  Please consider optimizing and modernizing as the project matures.

You will also see that much of the code is copied or the ideas copied from GNU Radio - not always built up from scratch.

## Folders

Since newsched is designed with modularity in mind, the key components that will be modular are pulled away from the runtime (now called the `gr`) directory

- gr - core components used by other modules (formerly `gnuradio-runtime`)
- blocklib - block library - already modular in GNU radio but pulled away from runtime
- schedulers - implementations of in-tree cpu and common domain schedulers
- docs - all general documentation related things
- bench - benchmarking flowgraphs 
- test - qa tests (some are in each module)

```
├── blocklib
├── bench
├── docs
├── gr
├── schedulers
└── test
```

[--> Next: Ports](03_Ports)