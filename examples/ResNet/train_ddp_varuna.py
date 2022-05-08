import os
import argparse
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
import signal

from varuna import Varuna

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

parser.add_argument('--rank', default=-1, type=int, help='node rank for distributed training')
parser.add_argument('--chunk_size', default=-1, type=int, help='micro batch size')
parser.add_argument('--batch-size', default=-1, type=int, help='per process batch size')
parser.add_argument('--stage_to_rank_map', default=None, type=str, help='stage to rank map of Varuna model')
parser.add_argument('--local_rank', default=-1, type=int, help='process rank in the local node')
parser.add_argument('--world-size', default=-1, type=int, help='number of nodes for distributed training')

class ResNet_Varuna(torch.nn.Module):
    def __init__(self):

        super(ResNet_Varuna, self).__init__()
        self.model = models.__dict__['resnet50'](num_classes=10)
        self.metric_fn = F.cross_entropy

    def forward(self,inputs, target):
        output = self.model(inputs)
        loss = self.metric_fn(output, target)
        return loss

def setup(backend, dist_url, rank, world_size):

    # initialize the process group
    dist.init_process_group(backend=backend, init_method=dist_url, rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def varuna_train(args): # how to set batch size, chunk size?

    num_epochs = 1
    print("rank is: ", args.rank, ", world size is: ", args.world_size)
    dist_url = "env://"
    dist_backend = "gloo"
    
    setup(dist_backend, dist_url, args.rank, args.world_size)

    print("batch size is: ", args.batch_size, " chunk size is: ", args.chunk_size)

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
                    train_dataset, num_replicas=args.world_size, rank=args.rank)

    print("------ Train loader will be defined with batch size: ", args.batch_size, " and world size: ", args.world_size)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None), 
        num_workers=8, pin_memory=True, sampler=train_sampler)

    print("Configure Varuna Model")

    model = ResNet_Varuna()

    def get_batch_fn(size, device=None): # is the "size" equal to chunk size or batch size?
            loader_ = torch.utils.data.DataLoader(train_dataset, batch_size=size) # what about sampling?
            image, target  = next(iter(loader_))
            inputs = {"inputs": image, "target": target}
            if device is not None:
                inputs["inputs"] = inputs["inputs"].to(device)
                inputs["target"] = inputs["target"].to(device)
            print(inputs["inputs"].size(), inputs["target"].size())
            return inputs
        
    model = Varuna(model, args.stage_to_rank_map, get_batch_fn, 
                    args.batch_size, args.chunk_size, fp16=False, 
                    local_rank=args.local_rank, device=-1) # how are these set?


    optimizer = torch.optim.SGD(model.parameters(), lr=0.1,
                                momentum=0.9,
                                weight_decay=1e-4)
    model.set_optimizer(optimizer)


    print("start training")

    
    for epoch in range(num_epochs):

        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        print("Start epoch: ", epoch)
        start = time.time()
        start_iter = time.time()
        
        ############################################################### VARUNA ##################################################

        for i, (images, target) in enumerate(train_loader):


            batch = {"inputs": images.to(model.device), "target": target.to(model.device)}
            #print(images, target)
            loss, overflow, grad_norm = model.step(batch)

            optimizer.step()
            optimizer.zero_grad()

            # measure elapsed time
            print(f"---- From worker with rank: {args.rank}, Iter {i} took {time.time()-start_iter}")
            start_iter = time.time()

        print(f"---- From worker with rank: {args.rank}, Epoch took: {time.time()-start}")
        model.reset_meas()

if __name__ == "__main__":

    args = parser.parse_args()
    
    def handler(signum,_):
        print("Got a signal - do nothing for now, just exit")
        exit()

    signal.signal(signal.SIGUSR1, handler)
    
    varuna_train(args)                                     
