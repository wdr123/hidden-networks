#!/bin/bash
base_job_name="LTH_15Dec"
job_file="the_job.sh"
identifier_name="hidden"
dir="op_"$identifier_name
mkdir -p $dir


array=(0 0.1 0.3 0.5 0.7 0.9)
for prune in "${array[@]}"
do
  export prune
  export arch="$1" dataset="$2"
  job_name=$base_job_name-$arch-$dataset-"${prune}"
  out_file=$dir/$job_name.out
  error_file=$dir/$job_name.err

  echo "prune_rate=${prune}" $arch $dataset
  sbatch -J $job_name -o $out_file -e $error_file $job_file
done
