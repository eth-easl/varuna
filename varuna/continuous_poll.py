import socket
from datetime import datetime
import os
import time
import sys
from random import randint
import subprocess

# cluster = sys.argv[1]
counter = 0
possibly_dead_nodes = []

available_machines_list = sys.argv[1]
running_machines_list = sys.argv[2]

server_ip = sys.argv[3]
server_port = int(sys.argv[4])

pattern = 'varuna-worker'
zone = 'europe-west4-a'

def process_response(res, pattern):

    instance_list = []
    reslines = res.split("\n")[1:]

    for r in reslines:
        info = r.split(" ")
        name = info[0]

        if (pattern in name):
            instance_list.append(name)
    return instance_list


def get_vm_list(pattern, zone):

    command = "gcloud compute instances list"
    cmd = command.split()
    result = subprocess.run(cmd, stdout=subprocess.PIPE).stdout.decode()
    instance_list = process_response(result, pattern)
    return instance_list

def check_VM_running(name, zone):
    command = "gcloud compute instances describe "  + name + " --zone " + zone
    result = subprocess.run(command.split(), stdout=subprocess.PIPE).stdout.decode()
    if "RUNNING" in result:
        return attempt_ssh(name, zone)
    return False

def attempt_ssh(name, zone):
    command = "gcloud compute ssh " + name + " --zone=" + zone + " -- -o ConnectTimeout=2 exit"
    print("ssh command is: ", command)
    result = os.system(command)
    if result==0:
        return True
    return False

def client(ip, port, message):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.connect((ip, port))
        sock.sendall(bytes(message, 'ascii'))

def poll_and_update():
    print(str(datetime.now()), flush=True)
    current_machines = get_current_machines()
    current_num_machines = len(current_machines)
    print("Current:", current_machines,flush=True)

    new_machines = get_available_machines()
    print("New", new_machines, flush=True)

    if sorted(new_machines) == sorted(current_machines):
        print("no morph", flush=True)
    else:
        # machines_added = [m for m in new_machines if m not in current_machines]
        msg = f"morph {len(new_machines)}"
        client(server_ip, server_port, msg)
        print(len(new_machines), flush=True)


def get_current_machines():
    f = open(running_machines_list,"r")
    machines = f.read().split("\n")
    machines = [m for m in machines if len(m) > 0]
    return machines

def get_available_machines():
    vm_list = get_vm_list(pattern, zone)
    print(vm_list)

    avail_vm_list = []

    for x in vm_list:
        if check_VM_running(x, zone):
            cmd =  "gcloud compute instances describe " + x + " --format get(networkInterfaces[0].networkIP) --zone " + zone
            ip_addr = subprocess.run(cmd.split(), stdout=subprocess.PIPE).stdout.decode()
            print(ip_addr)
            avail_vm_list.append(ip_addr)

    print(avail_vm_list)
    f = open(available_machines_list, 'w')
    for v in avail_vm_list: # TODO: what if someone is reading from this file?
        f.write(v)
    f.flush()

    return avail_vm_list

def notify():
     while True:
        poll_and_update()
        time.sleep(10)

if __name__ == "__main__":
    notify()
