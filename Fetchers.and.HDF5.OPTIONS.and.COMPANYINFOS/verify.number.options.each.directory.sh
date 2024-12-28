#!/bin/bash
# nice way to count anything in the lines after the matching directory ( this case aig ) 
#find ./ -type d -iname '*options_data' -exec sh -c 'echo "$(ls -la "$1" | wc -l) : $1"' sh {} \; | sed -n '/aig\/options_data/,$p'
#
find ./ -type d -iname '*options_data' -exec sh -c 'echo "$(ls -la "$1" | wc -l) : $1"' sh {} \; | 
sed -n '/angi\/options_data/,$p' | 
awk '{sum += $1; print} END {print "Total files: " sum-NR*2}'


