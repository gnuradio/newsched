# The First Test

We can now create a simple test case to try out our newly created blocks: `schedulers/st/test/qa_scheduler_st.cpp`

The testing framework used is [gtest](https://github.com/google/googletest), which has already been added into the `meson.build` as a dependency

The test demonstrates the basic commands for configuring and executing a flowgraph:

----
GTest macros encapsulate a test case by defining the test hierarchy and the test case name.  We have decided to create a series of tests under `SchedulerSTTest` and call this test `TwoSinks`, as we will be fanning out the output of one block to two sinks

```cpp
TEST(SchedulerSTTest, TwoSinks)
{
```
Declare the input data we will load into the `vector_source`, and based on the flowgraph, the data we expect to get out of the `vector_sink`

```cpp
    std::vector<float> input_data{ 1.0, 2.0, 3.0, 4.0, 5.0 };
    std::vector<float> expected_data;
    int k = 3.0;
    for (auto& inp : input_data)
    {
        expected_data.push_back(inp * k);
    }
```

Instantiate the blocks using the make functions
```cpp
    auto src = blocks::vector_source_f::make(input_data, false);
    auto op = blocks::multiply_const_ff::make(k);
    auto snk1 = blocks::vector_sink_f::make();
    auto snk2 = blocks::vector_sink_f::make();
```
Create the flowgraph object and connect our blocks
```cpp
    auto fg = flowgraph::make();
    fg->connect(src, 0, op, 0);
    fg->connect(op, 0, snk1, 0);
    fg->connect(op, 0, snk2, 0);
```

Create the scheduler, assign it to the flowgraph, and validate (finalize).  The scheduler can also be left out to use the default scheduler.

```cpp
    auto sched = schedulers::scheduler_st::make();
    fg->set_scheduler(sched);
    fg->validate();
```
Run the flowgraph
```cpp
    fg->start();
    fg->wait();
```
Check the results
```cpp
    EXPECT_EQ(snk1->data(), expected_data);
    EXPECT_EQ(snk2->data(), expected_data);
}
```

If all succeeded, we should see something like:

```
[0/1] Running all tests.
1/1 Single Threaded Scheduler Tests OK             0.12s


Ok:                 1   
Expected Fail:      0   
Fail:               0   
Unexpected Pass:    0   
Skipped:            0   
Timeout:            0   
```

There we have it - we ran our first flowgraph through newsched.  Now onto some more advanced topics