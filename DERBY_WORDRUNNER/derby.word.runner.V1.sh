#!/bin/bash

filename="$1"  # Get the filename from the command line argument

cat "$filename" | \
    tr '\t' ' ' | \
    tr -s ' ' | \
    tr ' ' '\n' | \
    awk '{print; system("sleep 0.1;clear")}'

