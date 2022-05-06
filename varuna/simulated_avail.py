from multiprocessing.connection import wait
import numpy as np
import random
import math
import os
import sys
import subprocess
import time

avail_machines_list = sys.argv[1]
zone = sys.argv[2]
pattern = sys.argv[3]

# TODO: put these in a "utils" file
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

def check_failures(p, rem):
    if rem <= 0:
        return 0
    r = random.random()
    if (r<=p):
        return random.randint(1, rem)
    return 0

def write_avail_machines(vlist, avail, filename):

    f = open(filename, 'w')
    n = len(vlist)
    for j in range(n):
        if avail[j]:
            f.write(vlist[j] + "\n")
    f.flush()


# p1: preemption probability/minute
# p2: rejoining probability/minute
# max_rem: maximum number of VMs to preempt per hour
# N: total number of VMs
# total_samples: the number of samples to check
# h: hours to train
def sim_avail(p1=0.0, p2=0.0, max_rem=0, N=1, vm_list=[]):

    alive = [True]*N # start with all machines alive
    killed = [-1.0]*N
    to_restart = [-1.0]*N

    failures = 0
    minutes = 0
    stride = 2*60 # how often to change availability
    restart = 6*60 # min to restart

    rem = max_rem

    while True: # while True here?
        # simulate availability and write to a file 
        if (minutes%60) == 0:
            # reset
            #print(max_rem-rem, " machines were preempted this past hour")
            rem = max_rem
        else:
            # 1. check for failures
            failed = check_failures(p1, rem) # number of machines that will fail
            rem -= failed
            failures += failed
            cnt = 0
            for j in range(N):
                if (cnt < failed) and alive[j]:
                    alive[j] = False
                    killed[j] = minutes
                    to_restart[j] = minutes+restart
                    cnt += 1
                elif (killed[j] > 0.0):
                    r = random.random()
                    print(j, r, to_restart[j], minutes)
                    if (r<=p2 and to_restart[j] <= minutes):
                        #print("--------- REJOIN! -------- at ", minutes)
                        alive[j] = True
                        killed[j] = -1.0

        al = alive.count(True)
        write_avail_machines(vm_list, alive, avail_machines_list)
        print(minutes, al, alive)
        minutes += stride
        time.sleep(stride)

# get the list of the available VMs and cluster size
vm_list = get_vm_list(pattern, zone)
num_vm = len(vm_list)
write_avail_machines(vm_list, [True]*num_vm, avail_machines_list)
sim_avail(p1=0.5, p2=0.1, max_rem=1, N=num_vm, vm_list=vm_list)

