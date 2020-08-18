#!/bin/bash

if [ "$#" -lt "1" ]
then
    FN=perf
else
    FN=$1
fi

echo "graph name: ${FN}"

if [ `mount | grep ramdisk | wc -l` -lt "1" ]
then
    sudo mount -t tmpfs -o rw,size=8G tmpfs /ramdisk
    echo "mounted ramdisk"
fi

sudo perf record -F 99 --call-graph fp -o /ramdisk/perf.dat -p $(pgrep -f build/run_opti_flowgraph) -- sleep 20

sudo perf script -i /ramdisk/perf.dat | /home/basti/src/FlameGraph/stackcollapse-perf.pl > ${FN}.fg

/home/basti/src/FlameGraph/flamegraph.pl ${FN}.fg > ${FN}.svg

