#!/bin/bash

# for file in *.tar.gz; do tar -xzf "$file"; done
for file in ./ingestion/pdf/*.tar.gz; do tar -xzf $file -C ./ingestion/pdf/; done
rm ./ingestion/pdf/*.tar.gz
mv ./ingestion/pdf/*/*.pdf ./ingestion/pdf/
# rm -r ./ingestion/pdf/*
find ./ingestion/pdf -mindepth 1 -maxdepth 1 -type d -exec rm -r {} +