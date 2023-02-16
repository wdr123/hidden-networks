#!/bin/bash
base_job_name="LTH_adaboost"
job_file="the_job_adaboost.sh"
identifier_name="adaboost"
dir="op_"$identifier_name
mkdir -p $dir



seed_list=(1 2 3)


for seed in "${seed_list[@]}";
do
  export seed
  job_name=$base_job_name-"resnet101_cifar100_0.15"
  out_file=$dir/$job_name.out
  error_file=$dir/$job_name.err

  echo $seed
  sbatch -J $job_name -o $out_file -e $error_file $job_file
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