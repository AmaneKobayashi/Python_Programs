#!/bin/bash
#PBS -l nodes=1:ppn=14
#PBS -e 54512-0/cheetah.stderr
#PBS -o 54512-0/cheetah.stdout
#PBS -N 54512-0
#PBS -q bl2-occupancy

cd $PBS_O_WORKDIR/54512-0

# This is the master job for runid, runid-light, runid-0.
# Subjobs must be submitted separatedly.

#if [ -e job.id ]; then
#   exit
#fi

echo $PBS_JOBID > job.id
hostname > job.host
source /home/sacla_sfx_app/setup.sh
ShowRunInfo -b 2 -r 54512 > run.info
/home/sacla_sfx_app/local/bin/prepare-cheetah-sacla-api2.py 54512 --bl=2 --clen=161.0 2>&1 >> cheetah.log
grep Error status.txt
if [ $? -eq 0 ]; then # Found
   for i in 1 2; do
      ln -s ../54512-0/status.txt ../54512-$i/
   done
   exit
fi

ln -s 54512-geom.h5 sacla-geom.h5
ln -s 54512-dark.h5 sacla-dark.h5

for i in 1 2; do
   if [ ! -e ../54512-$i/metadata.h5 ]; then
      cp 54512.h5 ../54512-$i/
   fi
   ln -s ../54512-0/54512-geom.h5 ../54512-$i/sacla-geom.h5
   ln -s ../54512-0/54512.geom ../54512-$i/54512-$i.geom
   ln -s ../54512-0/run.info ../54512-$i/run.info
   ln -s ../54512-0/54512-dark.h5 ../54512-$i/sacla-dark.h5
done

if [ ! -e run54512-0.h5 ]; then
   cp 54512.h5 run54512-0.h5
fi

/home/sacla_sfx_app/local/bin/cheetah-sacla-api2 --ini ../sacla-photon.ini --run 54512 -o run54512-0.h5 --bl 2  --type=0  2>&1 >> cheetah.log
rm 54512.h5

# th 100 gr 5000000 for > 10 keV
/home/sacla_sfx_app/local/bin/indexamajig -g 54512.geom --indexing=dirax --peaks=zaef --threshold=400 --min-gradient=10000 --min-snr=5 --int-radius=3,4,7 -o 54512-0.stream -j 14 -i -   <<EOF
run54512-0.h5
EOF
rm -fr indexamajig.*
grep Cell 54512-0.stream | wc -l > indexed.cnt
ruby /home/sacla_sfx_app/packages/tools/parse_stream.rb < 54512-0.stream > 54512-0.csv

rm job.id job.host
