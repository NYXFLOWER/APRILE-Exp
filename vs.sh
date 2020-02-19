#!/usr/bin/env bash

# Basic range in for loop
for value in {509..860}
do
    echo $value
    python run.py all all $value 2 0.99
done

echo All done