#!/bin/bash

kill $(ps aux | grep "train_ddp_varuna".py | grep -v grep | awk '{print $2}')