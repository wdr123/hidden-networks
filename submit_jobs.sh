#!/bin/bash
base_job_name="LTH_baselearner"
job_file="the_job.sh"
job_file1="the_job1.sh"
identifier_name="egde"
dir="op_"$identifier_name
mkdir -p $dir

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
  #        export arch="$1" dataset="$2"
          job_name=$base_job_name-$arch-$init-$dataset-"${prune}"
          out_file=$dir/$job_name.out
          error_file=$dir/$job_name.err

          echo "prune_rate=${prune}" $arch $dataset $init
          sbatch -J $job_name -o $out_file -e $error_file $job_file
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
  #        export arch="$1" dataset="$2"
        job_name=$base_job_name-$arch-standard-$dataset-"${prune}"
        out_file=$dir/$job_name.out
        error_file=$dir/$job_name.err

        echo "prune_rate=${prune}" $arch $dataset "standard"
        sbatch -J $job_name -o $out_file -e $error_file $job_file1
      done
    done
done