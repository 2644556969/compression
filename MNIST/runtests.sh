#!/bin/bash 

DIRECTORY="./results"

FILENAME=$1 




for i in {1..10}
do 
	#echo "Print $i" > "./$DIRECTORY/$FILENAME$i"
	python Compressnets/train.py --epochs=20 --batch_size=128 > "./$DIRECTORY/$FILENAME$i" & 
done 

