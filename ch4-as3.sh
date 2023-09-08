#!/bin/bash

ARRAY1=(0 1 3 5 7 11 13) # The different seeds
ARRAY2=(0 100 200 300 400 500 600 700 800 900) # The different thresholds

for i in "${ARRAY1[@]}"
do
	for j in "${ARRAY2[@]}"
	do
        jbsub -mem 2g -proj ch4-as3 -name part.$i.$j python3 when_is_zne_useful.py CH4_sto-3g_BK_grnd_AS4 $i $j 
	done
done 

wait
