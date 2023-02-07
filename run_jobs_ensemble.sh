#!/bin/bash

job_file="the_job.sh"
job_file1="the_job1.sh"

ensemble="True"
ensemble_subnet_init="None"
#ensemble_subnet_init="unsigned_constant signed_constant kaiming_normal kaiming_uniform standard"
data_repo="CIFAR10 CIFAR100"
arch_repo="resnet18 resnet50"
weight_kept=(0.03)
#weight_kept=(0.03 0.02 0.04 0.06)
# weight_kept1=(0.05 0.1 0.2 0.3)

for prune in "${weight_kept[@]}";
do
  for dataset in $data_repo;
    do
    for arch in $arch_repo;
      do
        export prune ensemble_subnet_init arch dataset ensemble

        echo "ensemble=${ensemble},ensemble_subnet_init=${ensemble_subnet_init}" "prune_rate=${prune}" $arch $dataset
        bash $job_file
      done
    done
done

