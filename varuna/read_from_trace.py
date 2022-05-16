from multiprocessing.connection import wait
import numpy as np
import os
import sys
import subprocess
import time

stride=int(sys.argv[1])
path=sys.argv[2]
pattern = sys.argv[3]
zone = sys.argv[4]
avail_machines_list = sys.argv[5]

gpus_per_VM=4


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

def write_avail_machines(vlist, avail, filename):

    f = open(filename, 'w')
    n = len(vlist)
    for j in range(n):
        if avail[j]:
            f.write(vlist[j] + "\n")
    f.flush()


def get_gpu_trace(path, stride):
    # get num of GPUs
    f = open(path, 'r')
    lines = f.readlines()
    total = len(lines)
    gpus=[]

    j=0
    t=0
    while j<total:
    # read <stride> lines
        i = 0
        running = 0
        while i<stride and j<total:
            l = lines[j]
            tokens = l.split()
            if ('RUNNING' in tokens):
                num_gpus = int(tokens[2].split("-")[4])
                running+=num_gpus
            i+=1
            j+=1
  
        gpus.append(running)

    return gpus

def sim_avail(vm_list, N, gpus_per_VM, avail_list, avail_machines_list):
    # config availability

    alive = [True]*N # start with all machines alive
    total_gpus = N*gpus_per_VM
    trace_len = len(avail_list)
    sleep_time = 60 # how often to change availability
    i=0
    while i<trace_len:
        cur_avail = avail_list[i]
        # cause fails
        to_rem = total_gpus-cur_avail
        j=0
        while j*4<to_rem:
            alive[j]=False
            j+=1
        
        # rest are alive
        while j<N:
            alive[j]=True
            j+=1

        write_avail_machines(vm_list, alive, avail_machines_list)
        i += 1
        time.sleep(sleep_time)
    

vm_list = get_vm_list(pattern, zone)
gpu_trace = get_gpu_trace(path, stride)
print(gpu_trace)
sim_avail(vm_list, len(vm_list), gpus_per_VM, gpu_trace, avail_machines_list)





