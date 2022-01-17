#!/usr/bin/env python3

from newsched import gr_unittest, gr, blocks
from newsched.schedulers import nbt


class test_basic(gr_unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    # def test_basic(self):
    #     nsamples = 100000
    #     input_data = list(range(nsamples))

    #     src = blocks.vector_source_f(input_data, False)
    #     cp1 = blocks.copy(gr.sizeof_float)
    #     cp2 = blocks.copy(gr.sizeof_float)
    #     snk1 = blocks.vector_sink_f()
    #     hd = blocks.head(len(input_data))

    #     fg1 = gr.flowgraph("fg1")
    #     fg2 = gr.flowgraph("fg2")

    #     fg1.connect(src, 0, cp1, 0)

    #     # Without the head - the other flowgraph has no mechanism for knowing when to stop
    #     # fg2.connect(cp2, 0, snk1, 0)
    #     fg2.connect(cp2, 0, hd, 0)
    #     fg2.connect(hd, 0, snk1, 0)

    #     # Fake the edge
    #     edge1 = gr.edge(cp1, cp1.get_port(0, gr.port_type_t.STREAM, gr.port_direction_t.OUTPUT),
    #                     None, None)
    #     edge1.set_custom_buffer(gr.buffer_net_zmq_properties.make("127.0.0.1", 1234))
    #     edge2 = gr.edge(None, None, cp2, cp2.get_port(0, gr.port_type_t.STREAM, gr.port_direction_t.INPUT)
    #                     )
    #     edge2.set_custom_buffer(gr.buffer_net_zmq_properties.make("127.0.0.1", 1234))


    #     fg1.add_edge(edge1)
    #     fg2.add_edge(edge2)

    #     fg1.start()
    #     fg2.start()
    #     fg1.wait()
    #     fg2.wait()

    #     self.assertEqual(input_data, snk1.data())


    # def test_nohead(self):
    #     nsamples = 100000
    #     input_data = list(range(nsamples))

    #     src = blocks.vector_source_f(input_data, False)
    #     cp1 = blocks.copy(gr.sizeof_float)
    #     cp2 = blocks.copy(gr.sizeof_float)
    #     snk1 = blocks.vector_sink_f()

    #     fg1 = gr.flowgraph("fg1")
    #     fg2 = gr.flowgraph("fg2")

    #     fg1.connect(src, 0, cp1, 0)
    #     fg2.connect(cp2, 0, snk1, 0)

    #     # Fake the edge
    #     edge1 = gr.edge(cp1, cp1.get_port(0, gr.port_type_t.STREAM, gr.port_direction_t.OUTPUT),
    #                     None, None)
    #     edge1.set_custom_buffer(gr.buffer_net_zmq_properties.make("127.0.0.1", 1234))
    #     edge2 = gr.edge(None, None, cp2, cp2.get_port(0, gr.port_type_t.STREAM, gr.port_direction_t.INPUT)
    #                     )
    #     edge2.set_custom_buffer(gr.buffer_net_zmq_properties.make("127.0.0.1", 1234))


    #     fg1.add_edge(edge1)
    #     fg2.add_edge(edge2)


    #     proxy1 = gr.fgm_proxy("127.0.0.1", 55122, True) 
    #     proxy2 = gr.fgm_proxy("127.0.0.1", 55122, False)
        
    #     sched1 = nbt.scheduler_nbt("nbtsched_host")
    #     sched2 = nbt.scheduler_nbt("nbtsched_remote")

    #     fg1.add_scheduler(sched1)
    #     fg1.add_fgm_proxy(proxy1)
    #     fg2.add_scheduler(sched2)
    #     fg2.add_fgm_proxy(proxy2)

    #     fg1.start()
    #     fg2.start()
    #     fg1.wait()
    #     fg2.wait()
 
    #     print(len(snk1.data()))
        
    #     # self.assertEqual(input_data, snk1.data())

    def test_moreintegrated(self):
        nsamples = 100000
        input_data = list(range(nsamples))

        src = blocks.vector_source_f(input_data, False)
        cp1 = blocks.copy(gr.sizeof_float)
        cp2 = blocks.copy(gr.sizeof_float)
        snk = blocks.vector_sink_f()

        fg1 = gr.flowgraph("FG On Local Host")

        fg1.connect(src, 0, cp1, 0)
        fg1.connect(cp1, 0, cp2, 0) # Implicitly a domain crossing
        fg1.connect(cp2, 0, snk, 0)

        # Indicate which scheduler to use in each domain
        sched1 = nbt.scheduler_nbt("nbtsched_host")
        sched2 = nbt.scheduler_nbt("nbtsched_remote")
        domain_confs = [
            gr.domain_conf(sched1, [src, cp1]),
            gr.domain_conf(sched2, [cp2, snk], gr.execution_host_properties("127.0.0.1", 8000))
        ]

        fg1.partition(domain_confs)

        fg1.start()
        fg1.wait()

        # Will need to get the data serialized from the remote host
        rx_data = snk.data()
        
        self.assertEqual(input_data, rx_data)

if __name__ == "__main__":
    gr_unittest.run(test_basic)
