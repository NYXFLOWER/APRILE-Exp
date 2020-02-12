#!/usr/bin/env bash

salloc -p gpu --gres gpu:1 -c 10 --mem 40g -t 8:0:0
