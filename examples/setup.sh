gcloud compute instance-templates create varuna-template-test-1 \
    --machine-type n1-standard-8 \
    --boot-disk-size 300GB \
    --accelerator type=nvidia-tesla-v100,count=1 \
    --image image-varuna-1 \
    --image-project ml-elasticity \
    --maintenance-policy TERMINATE \
    --restart-on-failure \
    --preemptible


gcloud compute instance-groups managed create varuna-test \
    --template varuna-template-test-1 --base-instance-name varuna-worker \
    --size 3 --zones europe-west4-a