#!/bin/bash
base_job_name="LTH_15Dec"
job_file="the_job.sh"
identifier_name="hidden"
dir="op_"$identifier_name
mkdir -p $dir


array=(0 0.1 0.3 0.5 0.7 0.9)
for i in "${array[@]}"
do
  export i
  export arch="$1" dataset="$2"
  job_name=$base_job_name-$((first))-$((second))-$((i))
  out_file=$dir/$job_name.out
  error_file=$dir/$job_name.err

  echo "prune_rate=${i}" $arch $dataset
  sbatch -J $job_name -o $out_file -t 1-00:00:00 -p tier3 -e $error_file $job_file
done
