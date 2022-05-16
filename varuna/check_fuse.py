import os

def check_and_fuse():

    f = open('/etc/mtab')
    lines = f.readlines()

    for l in lines:
        tokens = l.split()
        if 'varuna-checkpoints' in tokens:
            return

    print("FUSE")
    os.system("gcsfuse --implicit-dirs varuna-checkpoints globals")

check_and_fuse()