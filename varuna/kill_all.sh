ip_file=$1
machines=($(cat $ip_file))
nservers=${#machines[@]}

i=0
while [ $i -lt $nservers ]
do
    echo $i ${machines[i]}
    #ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 ${machines[i]} "sudo pkill -f varuna.launcher" 
    echo "try to kill"
    ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 ${machines[i]} "bash /home/fot/varuna/varuna/kill_local.sh"

    i=$(($i+1))
done