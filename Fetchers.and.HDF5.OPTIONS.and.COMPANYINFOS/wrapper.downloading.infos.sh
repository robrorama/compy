#!/bin/bash
for i in $( seq 1 1000 ) ;do echo processing iteration $i :: sleeping for 60 seconds $(date) ; sleep 60 ; time python3 003.new.dataframe.version.V2.staggers.py Tickers.csv ;done
