from newsched import blocks
from newsched.math.numpy.add import add_ff
from newsched import gr

tb = gr.flowgraph()
src0 = blocks.vector_source_f([1, 3, 5, 7, 9], False)
src1 = blocks.vector_source_f([0, 2, 4, 6, 8], False)
adder = add_ff()
sink = blocks.vector_sink_f()
tb.connect((src0, 0), (adder, 0))
tb.connect((src1, 0), (adder, 1))
tb.connect(adder, sink)
tb.run()

print(sink.data())
