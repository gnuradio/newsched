#!/usr/bin/env python3

from newsched import gr_unittest, gr, blocks
from newsched.schedulers import mt

class test_basic(gr_unittest.TestCase):

    def setUp(self):
        self.tb = gr.flowgraph()

    def tearDown(self):
        self.tb = None

    def test_basic(self):
        nsamples = 100000
        input_data = list(range(nsamples))

        limiting_factor = 32
        max_expected1 = limiting_factor * (nsamples // limiting_factor)
        limiting_factor = 64
        max_expected2 = limiting_factor * (nsamples // limiting_factor)
        expected_data1 = input_data[0:max_expected1]
        expected_data1 = [pow(2.0,5.0)*x for x in expected_data1]
        expected_data2 = input_data[0:max_expected2]
        expected_data2 = [pow(2.0,5.0)*x for x in expected_data2]


        src = blocks.vector_source_f(input_data, False)
        mult1 = blocks.multiply_const_ff(2.0, 1)
        mult2 = blocks.multiply_const_ff(2.0, 32)
        mult3 = blocks.multiply_const_ff(2.0, 32)
        mult4 = blocks.multiply_const_ff(2.0, 16)
        mult5 = blocks.multiply_const_ff(2.0, 4)
        mult6 = blocks.multiply_const_ff(2.0, 2)
        mult7 = blocks.multiply_const_ff(2.0, 64)

        snk1 = blocks.vector_sink_f()
        snk2 = blocks.vector_sink_f()

        self.tb.connect(src, 0, mult1, 0)
        self.tb.connect(mult1, 0, mult2, 0)
        self.tb.connect(mult2, 0, mult3, 0)
        self.tb.connect(mult3, 0, mult4, 0)
        self.tb.connect(mult3, 0, mult5, 0)
        self.tb.connect(mult4, 0, mult6, 0)
        self.tb.connect(mult5, 0, mult7, 0)
        self.tb.connect(mult6, 0, snk1, 0)
        self.tb.connect(mult7, 0, snk2, 0)

        self.tb.start()
        self.tb.wait()

        self.assertFloatTuplesAlmostEqual(expected_data1, snk1.data())
        self.assertFloatTuplesAlmostEqual(expected_data2, snk2.data())

if __name__ == "__main__":
    gr_unittest.run(test_basic)