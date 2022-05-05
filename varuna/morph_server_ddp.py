# to be run in manager
import socket
import threading
from threading import Thread
import socketserver
import time
from datetime import datetime
import os
import subprocess
import sys

is_morphing = False
available_machines_list = sys.argv[1]
running_machines_list = sys.argv[2]     # TODO: check race confitions here
PORT = int(sys.argv[3])
my_ip = socket.gethostbyname(socket.gethostname())
HOST = my_ip

is_restarting = False
is_morphing = False
is_preempting = False
checkpointed=False

class Handler(socketserver.BaseRequestHandler):

     triggermorph = threading.Lock()
     scripts_folder = os.path.dirname(os.path.abspath(__file__))

     def send_signal():
        print("sending signal", flush = True)
        sh = os.path.join(Handler.scripts_folder, "send_signal.sh")
        print(sh)
        p = None
        try:
            p = subprocess.call(['bash', sh, running_machines_list], timeout=10)
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError) as e:
            print("signal timed/errored out: ",e)
            if p is not None:
                p.kill()

     @staticmethod
     def start_remote(resume=-1):
        global available_machines_list, my_ip
        print("restarting", resume, flush=True)
        cmd = "python -m varuna.run_varuna --resume " + \
             f"--machine_list {available_machines_list} --manager_ip {my_ip}"
        os.system(cmd)


     def handle(self):
        global checkpointed, is_preempting, is_restarting, is_morphing
        data = str(self.request.recv(1024), 'ascii')
        cur_thread = threading.current_thread()
        recv_time = datetime.now()
        print("{} got something from {}: {}".format(recv_time, self.client_address, data), flush=True)

        if 'morph' in data:
            Handler.triggermorph.acquire()
            print("Lock acquired by morph:", is_restarting, is_morphing, is_preempting, flush=True)
            try:
                if not is_preempting and not is_restarting and not is_morphing:
                    print("Morphing!",flush=True)
                    is_restarting = True
                    is_morphing = True

                    response = Handler.send_signal()

                    # TODO: get new config and restart
                    # Handler.start_remote(checkpointed) # resume

                    is_morphing = False
                    is_restarting = False
                else:
                    print("morph change was already detected", is_morphing, is_preempting, is_restarting)
            except Exception as e:
                print("Caught exception while morphing:", e)
                is_morphing = False
                is_restarting = False
            Handler.triggermorph.release()
            print("Lock released by morph:", is_restarting, is_morphing, is_preempting)

        else:
            print("Not supported message")

class ThreadedTCPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):

    pass

if __name__ == "__main__":
    server = ThreadedTCPServer((HOST, PORT), Handler)

    print("Started server with IP: ", my_ip, ", host is: ", HOST)

    with server:
        server.serve_forever()

