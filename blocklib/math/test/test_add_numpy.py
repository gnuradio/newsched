from newsched import blocks
from newsched.math.numpy import add_ff, multiply_const_ff
from newsched import gr

tb = gr.flowgraph()
src0 = blocks.vector_source_f([1, 3, 5, 7, 9], False)
src1 = blocks.vector_source_f([0, 2, 4, 6, 8], False)
adder = add_ff()
mult = multiply_const_ff(3)
sink = blocks.vector_sink_f()
tb.connect((src0, 0), (adder, 0))
tb.connect((src1, 0), (adder, 1))
tb.connect([adder, mult, sink])
tb.run()

print(sink.data())
