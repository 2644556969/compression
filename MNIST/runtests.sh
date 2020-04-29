#!/bin/bash 

DIRECTORY="./results"

FILENAME=$1 




for i in {1..10}
do 
	#echo "Print $i" > "./$DIRECTORY/$FILENAME$i"
	
	python Compressnets/train_data_gen_labels.py --epochs=20 --batch_size=128 --logit_scale=.25  > "./$DIRECTORY/$FILENAME$i" & 
done 

