#!/bin/bash

server="34.73.155.73"
port=8097
nruns=3

# grab new logs
rm -rf logs/
cp -r /home/jcg6dn/supersonic/supersonic/logs/ .

# run visdom on last nruns
for run in $(ls logs | tail -$nruns); 
do
  echo
  echo $run
  python vis_plot.py -port $port -server $server -run $run
done;

