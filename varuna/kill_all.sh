ip_file=$1
machines=($(cat $ip_file))
nservers=${#machines[@]}

i=0
while [ $i -lt $nservers ]
do
    echo $i ${machines[i]}
    #ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 ${machines[i]} "sudo pkill -f varuna.launcher" 
    echo "try to kill"
    ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 ${machines[i]} "sudo kill $(ps aux | grep train_ddp_varuna.py | grep -v grep | awk '{print $2}')"

    i=$(($i+1))
done