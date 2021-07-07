#!/usr/bin/env python3

from newsched import gr_unittest, gr, blocks
import numpy as np


class test_basic(gr_unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_basic(self):
        mult = blocks.multiply_const_ff(1.0, 1)

        work_input = gr.block_work_input(np.ones((10,)))
        work_output = gr.block_work_output(np.zeros((10,)))

        mult.work([work_input], [work_output])

        input_vec = work_input.numpy()
        output_vec = work_output.numpy()

        self.assertTrue(np.allclose(input_vec, output_vec))


if __name__ == "__main__":
    gr_unittest.run(test_basic)
