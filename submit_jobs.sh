#!/bin/bash
base_job_name="LTH_newbase"
job_file="the_job.sh"
job_file1="the_job1.sh"
identifier_name="egde1"
dir="op_"$identifier_name
mkdir -p $dir


ensemble="False"
ensemble_subnet_init="kaiming_uniform kaiming_normal"
subnet_init="kaiming_uniform"
#subnet_init="kaiming_normal kaiming_uniform"
#subnet_init="unsigned_constant signed_constant kaiming_normal kaiming_uniform"
data_repo="CIFAR10 CIFAR100 TinyImageNet"
arch_repo="vit swin cResNet18 cResNet50 cResNet101"
#arch_repo="cResNet101"
weight_kept=(0.05 0.05 0.05 0.15)
#weight_kept1=(0.15 0.1 0.2 0.3)

for prune in "${weight_kept[@]}";
do
  for init in $subnet_init;
  do
    for dataset in $data_repo;
      do
      for arch in $arch_repo;
        do
          export prune init arch dataset ensemble
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


#for prune in "${weight_kept1[@]}";
#do
#  for dataset in $data_repo;
#    do
#      for arch in $arch_repo;
#      do
#        export prune arch dataset
#  #        export arch="$1" dataset="$2"
#        job_name=$base_job_name-$arch-standard-$dataset-"${prune}"
#        out_file=$dir/$job_name.out
#        error_file=$dir/$job_name.err
#
#        echo "prune_rate=${prune}" $arch $dataset "standard"
#        sbatch -J $job_name -o $out_file -e $error_file $job_file1
#      done
#    done
#done