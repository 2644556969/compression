#!/bin/bash 

directory="./results"

filename="trainresults"

num_tests=10 

if ["$1" != ""]; then 
	filename = $1 
fi

if ["$2" != ""]; then 
	num_tests = $2 
fi

for i in {1..$num_tests}
do 
	echo "Print $i" 
	echo "Print $directory/$filename$i"
	#python Compressnets/train.py --epochs=20 --num_
done 
