#!/bin/bash

conda init bash
conda activate qiskit_runtime_2

ARRAY1=(0 1 3) # The different seeds
ARRAY2=(0 150 300 700 1200 1800 inf) # The different thresholds
ARRAY3=(1 3 8) # The different noise scalings
ARRAY4=(CH4_sto-3g_BK_grnd_AS4 LiH_sto-3g_BK_1.45_AS4 H2O_sto-3g_BK_104_AS4 H2_6-31g_BK_0.7_AS4) #the different molecules

for i in "${ARRAY1[@]}"
do
	for j in "${ARRAY2[@]}"
	do
        for k in "${ARRAY3[@]}"
        do
            for l in "${ARRAY4[@]}"
            do
                jbsub -mem 2g -proj exxp-7 -name part.$i.$j.$k.$l python3 when_is_zne_useful_scale_noise.py $l $i $j $k
            done
        done
	done
done

wait

