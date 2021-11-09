# Sync Block

Now that we have all the pieces in place, let's create our most basic blocks, and get to the point where we can create a flowgraph in a qa test and validate the result

First we will look at the `sync_block` class which wraps `block` to ensure `ninputs == noutputs` for every work call

Reference [commit](https://github.com/gnuradio/newsched/commit/7f0c0648ebeaca374db03b67b3762633ad7cf6ed)

Just like in GR, the `sync_block` class guarantees the 1:1 sample input/output relationship.   In GR we had a separate `work` function for `sync_block`s that wrapped the `general_work` function of a `block`.  Since we have stuck with a common work function signature for _all_ blocks, extra checks must be completed by the `sync_block` class.  

The `sync_block` class performs checks on inputs and outputs before and after the call to the derived block's work function

The sync_block guarantees that the input and output buffers to the
work function of the derived block fit the constraints of the 1:1 sample input/output relationship by doing the following

1. Check all inputs and outputs have the same number of items
2. Fix all inputs and outputs to the absolute min across ports
3. Call the work() function on the derived block
4. Throw runtime_error if n_produced is not the same on every port
5. Set n_consumed = n_produced for every input port

