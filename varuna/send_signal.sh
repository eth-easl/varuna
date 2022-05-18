ip_file=$1
local_pid_path="/tmp/varuna/local_parent_pid"
machines=($(cat $ip_file))

echo "triggering stop signal"
i=0
while [ $i -lt ${#machines[@]} ]
do
    gcloud compute ssh ${machines[i]} --zone us-west1-a --command "kill -10 \$(cat $local_pid_path)"
    i=$(($i+1))
done
