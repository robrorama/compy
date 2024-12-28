find . -type d -exec sh -c 'echo $(find "{}" -maxdepth 1 -type f | wc -l) "{}"' \; | sort -n

