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

from varuna import Varuna


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

def varuna_train(local_rank, world_size, batch_size=32, stage_to_rank_map="0;", chunk_size=32, num_epochs=1): # how to set batch size, chunk size?

    print("local rank is: ", local_rank, ", world size is: ", world_size)
    dist_url = "env://"
    dist_backend = "gloo"
    
    setup(dist_backend, dist_url, local_rank, world_size)

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

    train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=(train_sampler is None), # ???????????????????
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
        
    model = Varuna(model, stage_to_rank_map, get_batch_fn, 
                    batch_size, chunk_size, fp16=False, 
                    local_rank=-1, device=-1) # how are these set?


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

            if i==10:
                break
            #if args.gpu is not None:
            #    images = images.cuda(args.gpu, non_blocking=True)
            #    target = target.cuda(args.gpu, non_blocking=True)

            batch = {"inputs": images.to(model.device), "target": target.to(model.device)}
            #print(images, target)
            loss, overflow, grad_norm = model.step(batch)
            print(loss)
            # measure accuracy and record loss
            # acc1, acc5 = accuracy(output, target, topk=(1, 5))
            #losses.update(loss if args.varuna else loss.item(), images.size(0))
            # top1.update(acc1[0], images.size(0))
            # top5.update(acc5[0], images.size(0))

            optimizer.step()
            optimizer.zero_grad()

            # measure elapsed time
            print("Iter ", i, " took ", time.time()-start_iter)
            start_iter = time.time()

        print("Epoch took: ", time.time()-start)

if __name__ == "__main__":

    world_size=1
    mp.set_start_method("spawn")
    mp.spawn(varuna_train,args=(world_size,),nprocs=world_size)                                      
