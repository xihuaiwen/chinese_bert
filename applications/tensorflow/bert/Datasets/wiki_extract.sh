#!/bin/bash

# Download the latest wikipedia dump and save it in the path given by the first argument
if [ $# -ne 2 ]; then
    echo "Usage: ${0} path-to-wikidump.xml path-to-destination"
    exit
fi

wikidump_path="${1}"
output_path="${2}"

python -m  wikiextractor.WikiExtractor "${wikidump_path}" -b 1000M --processes 16 --filter_disambig_pages -o "${output_path}" 
