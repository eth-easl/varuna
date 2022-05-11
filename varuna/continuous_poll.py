import socket
from datetime import datetime
import os
import time
import sys
from random import randint
import subprocess
import argparse

parser = argparse.ArgumentParser(description='Poll for updates and notify')

parser.add_argument('--server-ip', default="127.0.0.1", type=str, help='IP address of the morph server')
parser.add_argument('--server-port', default=1234, type=int, help='Port of the morph server')
parser.add_argument('--from-sim', default=False, action='store_true', help='Whether or not to get available VMs from simulator')
parser.add_argument('--zone', default="us-west1-a", type=str, help='Cluster zone')
parser.add_argument('--available_machines_list', default="", type=str, help='File of the available machine IPs')
parser.add_argument('--running_machines_list', default="", type=str, help='File of the currently running machine IPs')
parser.add_argument('--simulated_machines_list', default="", type=str, help='File of the simulated machine IPs')

# cluster = sys.argv[1]
counter = 0
possibly_dead_nodes = []

pattern = 'varuna-worker'
args = parser.parse_args()

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

    new_machines = []
    if args.from_sim:
        new_machines = get_available_machines_sim()
    else:
        new_machines = get_available_machines()
    print("New", new_machines, flush=True)

    if sorted(new_machines) == sorted(current_machines):
        print("no morph", flush=True)
    else:
        # machines_added = [m for m in new_machines if m not in current_machines]
        msg = f"morph {len(new_machines)}"
        client(args.server_ip, args.server_port, msg)
        print(len(new_machines), flush=True)


def get_current_machines():
    f = open(args.running_machines_list,"r")
    machines = f.read().split("\n")
    machines = [m for m in machines if len(m) > 0]
    return machines

def get_avail(vm_list):
    avail_vm_list = []

    for x in vm_list:
        if check_VM_running(x, args.zone):
            cmd =  "gcloud compute instances describe " + x + " --format get(networkInterfaces[0].networkIP) --zone " + args.zone
            ip_addr = subprocess.run(cmd.split(), stdout=subprocess.PIPE).stdout.decode()
            print(ip_addr)
            avail_vm_list.append(ip_addr)

    print(avail_vm_list)
    f = open(args.available_machines_list, 'w')
    for v in avail_vm_list: # TODO: what if someone is reading from this file?
        f.write(v)
    f.flush()
    return avail_vm_list


def get_available_machines_sim():
    f = open(args.simulated_machines_list,"r")
    machines = f.read().split("\n")
    machines = [m for m in machines if len(m) > 0]
    return get_avail(machines)

def get_available_machines():
    vm_list = get_vm_list(pattern, args.zone)
    print(vm_list)
    return get_avail(vm_list)

def notify():
     while True:
        poll_and_update()
        time.sleep(10)

if __name__ == "__main__":
    notify()
