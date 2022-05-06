# checks the number of available VMs (in 'running' mode)
import os
import sys
import time
import subprocess

pattern = 'varuna-worker'
zone = 'europe-west4-a'

avail_vm_file = sys.argv[1]

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

while True:
    vm_list = get_vm_list(pattern, zone)
    avail_vm_list = [x for x in vm_list if check_VM_running(x, zone)]
    print(avail_vm_list)
    f = open(avail_vm_file, 'w')
    for v in avail_vm_list:
        f.write(v + "\n")
    f.flush()
    time.sleep(10)
