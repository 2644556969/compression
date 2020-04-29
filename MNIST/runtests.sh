#!/bin/bash 

DIRECTORY="./results"

FILENAME=$1 




for i in {1..10}
do 
	#echo "Print $i" > "./$DIRECTORY/$FILENAME$i"
	
	python Compressnets/train_data_gen.py --epochs=25 --batch_size=128 --logit_scale=.7  > "./$DIRECTORY/$FILENAME$i" & 
done 

