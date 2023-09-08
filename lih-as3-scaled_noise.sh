#!/bin/bash

conda init bash
conda activate qiskit_runtime_2

ARRAY1=(0 1 3 5 7 11 13) # The different seeds
ARRAY2=(0 100 200 300 400 500 600 700 800 900) # The different thresholds

for i in "${ARRAY1[@]}"
do
	for j in "${ARRAY2[@]}"
	do
        jbsub -mem 2g -proj lih-as3-10x -name part.$i.$j python3 when_is_zne_useful_scale_noise.py LiH_sto-3g_BK_1.45_AS3 $i $j 5
	done
done 

wait
