from http.server import BaseHTTPRequestHandler,HTTPServer
import json
import numpy as np

from train import trainer
from utils import *

class RequestHandler(BaseHTTPRequestHandler):

    def __init__(self,req,client,server):
        BaseHTTPRequestHandler.__init__(self,req,client,server)
    
    def log_message(self, format, *args):
        #silent
        return
            
    def do_GET(self):
        
        request_path = self.path
        
        print("\ndo_Get it should not happen\n")
        
        self.send_response(200)
        
    def do_POST(self):

        _debug = False
        
        request_path = self.path
        
        request_headers = self.headers
        content_length = request_headers.get_all('content-length')
        length = int(content_length[0]) if content_length else 0
        content = self.rfile.read(length)

        if _debug:
            print("\n----- Request Start ----->\n")
            print(request_path)
            print(request_headers)
            print(content)
            print("<----- Request End -----\n")
        
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        self.wfile.write(self.dispatch(content.decode("ascii")).encode("ascii"))

    def get_target(self,msg):
        obj = json.loads(msg)
        return obj["state"]["side"] , obj

    def dispatch(self,msg):
        #print(msg)
        target , json_obj = self.get_target(msg)
        agent = dispatch_table[target]
        st = json_obj
        raw_act = agent.send((st["state"],float(st["reward"]),st["done"] == "true"))
        if target == "Radiant":
            sign = 1
        else:
            sign = -1
        ret = "%f %f"%(sign * raw_act[0], sign * raw_act[1])
        print(target,ret)
        return ret

    do_PUT = do_POST
    do_DELETE = do_GET

def start_env(params,shared_model,shared_grad_buffer):
    init_dispatch(params,shared_model,shared_grad_buffer)
    port = 8080
    print('Listening on localhost:%s' % port)
    server = HTTPServer(('', port), RequestHandler)
    server.serve_forever()

def parse_creep(sign,creep):
    count = 0
    _x = 0.0
    _y = 0.0

    for c in creep:
        count = count + 1
        _x = _x + c[5]
        _y = _y + c[6]
    
    
    
    _x = _x / count * sign
    _y = _y / count * sign
    if count == 1:
        count = 0
    return [_x, _y, count]

def parse_tup(side,raw_tup):
    origin = raw_tup
    raw_tup = raw_tup[0]
    self_input = raw_tup["self_input"]
    if side == "Radiant":
        sign = 1
        _side = 0
    else:
        sign = -1
        _side = 1
    ret = [_side,sign * self_input[11],sign * self_input[12]]
    ret = ret + parse_creep(sign,raw_tup["ally_input"])
    ret = ret + parse_creep(sign,raw_tup["enemy_input"])
    return (ret,origin[1],origin[2])

def gen_model():
    params,shared_model,shared_grad_buffers,side = yield

    while True:
        act = np.asarray([0.0,0.0])
        agent = trainer(params,shared_model,shared_grad_buffers)

        tup = yield (act[0] * 500,act[1] * 500)

        total_reward = 0.0

        agent.pre_train()

        tick = 0

        while True:
            tick += 1
            move_order = (act[0] * 500,act[1] * 500)

            tup = yield move_order

            tup = parse_tup(side, tup)

            print("origin input ", tup ,flush=True)

            total_reward += tup[1]

            act = get_action(agent.step(tup,None,0.0))


            if tup[2]:
                break

dispatch_table = {}

def init_dispatch(params,shared_model,shared_grad_buffer):
    dispatch_table["Dire"] = gen_model()
    dispatch_table["Dire"].send(None)
    dispatch_table["Dire"].send((params,shared_model,shared_grad_buffer,"Dire"))

    dispatch_table["Radiant"] = gen_model()
    dispatch_table["Radiant"].send(None)
    dispatch_table["Radiant"].send((params,shared_model,shared_grad_buffer,"Radiant"))
