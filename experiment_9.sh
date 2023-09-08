#!/bin/bash

conda init bash
conda activate qiskit_runtime_2

ARRAY1=(0 1 3 5) # The different seeds
ARRAY2=(0 inf) # The different thresholds
ARRAY3=(CH4_sto-3g_BK_grnd_AS3 CH4_sto-3g_BK_grnd_AS4 LiH_sto-3g_BK_1.45_AS3 LiH_sto-3g_BK_1.45_AS4 H2O_sto-3g_BK_104_AS3 H2O_sto-3g_BK_104_AS4 H2_6-31g_BK_0.7_AS3 H2_6-31g_BK_0.7_AS4) #the different molecules
#ARRAY3=(LiH_sto-3g_BK_1.45_AS4)

for i in "${ARRAY1[@]}"
do
	for j in "${ARRAY2[@]}"
	do
        for k in "${ARRAY3[@]}"
        do
            jbsub -mem 2g -proj exp-9 -name part.$i.$j.$k python3 dZNE-depol-noise.py $k $i $j
        done
	done
done 

wait

