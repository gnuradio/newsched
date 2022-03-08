import requests
import json
from gnuradio import gr
import random
import inspect

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

    def rpc_return(func):
        def inner(*args, **kwargs):
            r = func(*args, **kwargs)
            # print(r)
            # print(r.text)
            return json.loads(r.text)['result']
        return inner

    def rpc_execute(timeout=0.1):
        def wrapper(func):
            def inner(*args, **kwargs):
                json_payload = func(*args)
                if not json_payload:
                    argspec = inspect.getargspec(func) # get the args
                    obj = inspect.unwrap(func) # get the command
                    json_payload = {'command': obj.__name__}
                    for i, a in enumerate(args):
                        if i > 0:
                            json_payload[argspec.args[i]] = args[i]
                # print(json_payload) # TODO: make a verbose flag 
                return requests.post(args[0].url + '/execute', json=json_payload, timeout=timeout)

            return inner
        return wrapper

    @rpc_execute()
    def flowgraph_create(self, fg_name):
        pass

    @rpc_execute()
    def block_create(self, block_name, block):

        json_str = block.to_json()
        payload = json.loads(json_str)
        payload['command'] = 'block_create'
        payload['block_name'] = block_name
        return payload


    @rpc_execute()
    def block_create_params(self, block_name, params):

        payload = params
        payload['command'] = 'block_create'
        payload['block_name'] = block_name
        return payload

    @rpc_return
    @rpc_execute()
    def block_method(self, block_name, method, params):
        pass

    @rpc_return
    @rpc_execute()
    def block_parameter_query(self, block_name, parameter_name):
        pass

    @rpc_execute()
    def block_parameter_change(self, block_name, parameter_name, encoded_value):
        pass

    @rpc_return
    @rpc_execute()
    def flowgraph_connect(self, fg_name, src, dst, edge_name):
        pass

    @rpc_execute()
    def runtime_create(self, rt_name):
        pass

    @rpc_execute()
    def runtime_initialize(self, rt_name, fg_name):
        pass

    @rpc_execute()
    def runtime_start(self, rt_name):
        pass

    @rpc_execute(timeout=None)
    def runtime_wait(self, rt_name):
        pass

    @rpc_execute()
    def runtime_stop(self, rt_name):
        pass

    @rpc_return
    @rpc_execute()
    def runtime_create_proxy(self, rt_name, svr_port, upstream): #rt_name, payload):
        pass

    
    @rpc_execute()
    def runtime_connect_proxy(self, proxy_name, ipaddr, port): #proxy_name, payload):
       pass