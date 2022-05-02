import os
from platform import node
import sys
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torchvision import models, datasets, transforms
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn.functional as F
from torch.multiprocessing import Pool, Process, set_start_method, Manager, Value, Lock
from datetime import timedelta
import random
import numpy as np
import time

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '1234'

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def train(local_rank, world_size):

    print("local rank is: ", local_rank, ", world size is: ", world_size)
    setup(local_rank, world_size)

    torch.cuda.set_device(local_rank)
    model = models.__dict__['resnet50'](num_classes=10)
    model = model.to(local_rank) # to GPU

    model = DDP(model, device_ids=[local_rank])

    optimizer =  torch.optim.SGD(model.parameters(), lr=0.1)
    metric_fn = F.cross_entropy

    print("Configure dataset")

    train_dir = "/cifar/train"
    kwargs = {'num_workers': 8, 'pin_memory': True}

    train_dataset = \
            datasets.ImageFolder(train_dir,
                                    transform=transforms.Compose([
                                        transforms.RandomResizedCrop(224),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                            std=[0.2023, 0.1994, 0.2010])

                                        ]))

    train_sampler = torch.utils.data.distributed.DistributedSampler(
                    train_dataset, num_replicas=world_size, rank=local_rank)
    train_loader = torch.utils.data.DataLoader(
                    train_dataset, batch_size=128, sampler=train_sampler, num_workers=8)



    print("start training")

    for i in range(1):
        print("Start epoch: ", i)

        model.train()
        train_size = len(train_loader)
        print("train size is: ", train_size)

        train_iter = enumerate(train_loader)

        start = time.time()
        start_iter = time.time()

        batch_idx, batch = next(train_iter)

        while batch_idx < train_size:
            optimizer.zero_grad()
            data, target = batch[0].to(local_rank), batch[1].to(local_rank)
            output = model(data)
            loss = metric_fn(output, target)
            loss.backward()
            optimizer.step()
            print("Iter ", batch_idx, " took ", time.time()-start_iter)
            batch_idx, batch = next(train_iter)

            start_iter = time.time()

        print("Epoch took: ", time.time()-start)

if __name__ == "__main__":

    world_size=1
    mp.set_start_method("spawn")
    mp.spawn(train,args=(world_size,),nprocs=world_size,join=True)                                      
