from base64 import encode
import requests
import json
from gnuradio import gr
import pmtf

class client(gr.rpc_client_interface):
    def __init__(self, ipaddr, port, https = False) -> None:
        gr.gr_python.rpc_client_interface.__init__(self)
        self.set_pyblock_detail(gr.pyblock_detail(self))

        self.ipaddr = ipaddr
        self.port = port
        self.https = https
        self.url = 'http://' if not self.https else 'https://' 
        self.url += self.ipaddr + ':' + str(self.port)

        self.conn = None

    # def connect(self):
    #     url = 'http://' if not self.https else 'https://'
    #     url += self.ipaddr + ':' + self.port
    #     self.conn = http.client.HTTPConnection(url)

    # def check_connection(self):
    #     if not self.conn:
    #         raise Exception("Client has not yet been connected to server")

    # TODO: do this with decorator
    # def rpc_execute(func):
    #     payload = {'command': 'flowgraph_create', 'fg_name': fg_name}

    def flowgraph_create(self, fg_name):

        payload = {'command': 'flowgraph_create', 'fg_name': fg_name}
        r = requests.post(self.url + '/execute', json=payload, timeout=0.1)

        print(r)


    def block_create(self, block_name, block):

        json_str = block.to_json()
        payload = json.loads(json_str)
        payload['command'] = 'block_create'
        payload['block_name'] = block_name

        r = requests.post(self.url + '/execute', json=payload)

        print(r)

    def block_create_params(self, block_name, params):

        payload = params
        payload['command'] = 'block_create'
        payload['block_name'] = block_name

        r = requests.post(self.url + '/execute', json=payload)

        print(r)

    def block_method(self, block_name, method, params):
        
        payload = {}
        payload['command'] = 'block_method'
        payload['block_name'] = block_name
        payload['method'] = method
        payload['params'] = params

        r = requests.post(self.url + '/execute', json=payload)
        return json.loads(r.text)['result']
        print(r)

    def block_parameter_query(self, block_name, parameter_name):
        payload = {}
        payload['command'] = 'block_parameter_query'
        payload['block_name'] = block_name
        payload['parameter'] = parameter_name

        r = requests.post(self.url + '/execute', json=payload)
        pmt_b64 = json.loads(r.text)['result']
        return pmt_b64

    def block_parameter_change(self, block_name, parameter_name, encoded_value):
        payload = {}
        payload['command'] = 'block_parameter_change'
        payload['block_name'] = block_name
        payload['parameter'] = parameter_name
        payload['encoded_value'] = encoded_value

        r = requests.post(self.url + '/execute', json=payload)
        print(r)

    def flowgraph_connect(self, fg_name, src_block, src_port, dst_block, dst_port, edge_name):

        payload = {"command": "flowgraph_connect", "fg_name": fg_name}
        if src_block and src_port:
            payload["src"] = (src_block, src_port)

        if dst_block and dst_port:
            payload["dst"] = (dst_block, dst_port)

        if edge_name:
            payload["edge_name"] = edge_name

        r = requests.post(self.url + '/execute', json=payload)

        print(r)


    def runtime_initialize(self, fg_name, src_block, src_port, dst_block, dst_port, edge_name):

        payload = {"command": "flowgraph_connect", "fg_name": fg_name}
        if src_block and src_port:
            payload["src"] = (src_block, src_port)

        if dst_block and dst_port:
            payload["dst"] = (dst_block, dst_port)

        if edge_name:
            payload["edge_name"] = edge_name

        r = requests.post(self.url + '/execute', json=payload)

        print(r)



    def runtime_create(self, rt_name):
        payload = {"command": "runtime_create", "rt_name": rt_name}
        r = requests.post(self.url + '/execute', json=payload)

        print(r)

    def runtime_initialize(self, rt_name, fg_name):
        payload = {"command": "runtime_initialize", "rt_name": rt_name, "fg_name": fg_name}
        r = requests.post(self.url + '/execute', json=payload)

        print(r)

    def runtime_start(self, rt_name):
        payload = {"command": "runtime_start", "rt_name": rt_name}
        r = requests.post(self.url + '/execute', json=payload)

        print(r)

    def runtime_wait(self, rt_name):
        payload = {"command": "runtime_wait", "rt_name": rt_name}
        r = requests.post(self.url + '/execute', json=payload)

        print(r)

    def runtime_stop(self, rt_name):
        payload = {"command": "runtime_stop", "rt_name": rt_name}
        r = requests.post(self.url + '/execute', json=payload)

        print(r)