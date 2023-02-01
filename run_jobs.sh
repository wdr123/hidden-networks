#!/bin/bash

job_file="the_job.sh"
job_file1="the_job1.sh"


subnet_init="unsigned_constant signed_constant kaiming_normal kaiming_uniform"
data_repo="CIFAR10 CIFAR100"
arch_repo="resnet18 resnet50"
weight_kept=(0.01 0.02 0.04 0.06)
weight_kept1=(0.05 0.1 0.2 0.3)

for prune in "${weight_kept[@]}";
do
  for init in $subnet_init;
  do
    for dataset in $data_repo;
      do
        for arch in $arch_repo;
        do
          export prune init arch dataset

          echo "prune_rate=${prune}" $arch $dataset $init
          bash $job_file
        done
      done
  done
done


for prune in "${weight_kept1[@]}";
do
  for dataset in $data_repo;
    do
      for arch in $arch_repo;
      do
        export prune arch dataset

        echo "prune_rate=${prune}" $arch $dataset "standard"
        bash $job_file1
      done
    done
done