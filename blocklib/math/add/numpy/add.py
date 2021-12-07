from newsched import math
from newsched import gr

class add_ff(math.add_ff):
    def __init__(self, *args, **kwargs):
        math.add_ff.__init__(self, *args, **kwargs, impl = math.add_ff.available_impl.pyshell)
        self.set_py_handle(self)
    
    def work(self, inputs, outputs):
        noutput_items = outputs[0].n_items
        
        outputs[0].produce(noutput_items)

        inbuf1 = gr.get_input_array(self, inputs, 0)
        inbuf2 = gr.get_input_array(self, inputs, 1)
        outbuf1 = gr.get_output_array(self, outputs, 0)

        outbuf1[:] = inbuf1 + inbuf2

        return gr.work_return_t.WORK_OK 