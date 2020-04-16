#!/bin/bash 

DIRECTORY="./results"

FILENAME=$1 

NUMTESTS=$2



for i in {1..10}
do 
	echo "Print $i" 
	echo "Print $DIRECTORY/$FILENAME$i"
	#python Compressnets/train.py --epochs=20 --num_
done 



for i in {1.."$NUMTESTS"}
do 
	echo "Print $i" 
	echo "Print $DIRECTORY/$FILENAME$i"
	#python Compressnets/train.py --epochs=20 --num_
done 
