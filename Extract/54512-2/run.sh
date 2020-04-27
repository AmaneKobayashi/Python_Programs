#!/bin/bash
#PBS -l nodes=1:ppn=14
#PBS -e 54512-2/cheetah.stderr
#PBS -o 54512-2/cheetah.stdout
#PBS -N 54512-2
#PBS -q bl2-occupancy

cd $PBS_O_WORKDIR/54512-2/

#if [ -e job.id ]; then
#   exit
#fi

echo $PBS_JOBID > job.id
hostname > job.host
source /home/sacla_sfx_app/setup.sh

i=0
while :; do
   let i=i+1
   if [ -e sacla-dark.h5 ]; then
      break
   fi
   grep Error status.txt
   if [ $? -eq 0 ]; then
      exit
   fi

   if [ $i -gt 500 ]; then
      echo "Status: Status=Error-TimeoutWaitingDarkAverage" > status.txt
      exit -1
   fi

   sleep 2
done

cp 54512.h5 run54512-2.h5
/home/sacla_sfx_app/local/bin/cheetah-sacla-api2 --ini ../sacla-photon.ini --run 54512 -o run54512-2.h5 --bl 2 --type=2 2>&1 >> cheetah.log
rm 54512.h5

# th 100 gr 5000000 for > 10 keV
/home/sacla_sfx_app/local/bin/indexamajig -g 54512-2.geom --indexing=dirax --peaks=zaef --threshold=400 --min-gradient=10000 --min-snr=5 --int-radius=3,4,7 -o 54512-2.stream -j 14 -i -   <<EOF
run54512-2.h5
EOF
rm -fr indexamajig.*
grep Cell 54512-2.stream | wc -l > indexed.cnt
ruby /home/sacla_sfx_app/packages/tools/parse_stream.rb < 54512-2.stream > 54512-2.csv

rm job.id job.host
