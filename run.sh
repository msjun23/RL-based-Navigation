#!/bin/bash

# List of indices
test_list=(42 64 296 297 136 54 173 61 85 197 62 167 206 129 89 113 235 93 290 133 141 24 60 255 66 191 118 28 154 201 187 19 102 71 140 57 150 94 81 40 101 121 0 200 267 275 260 172 59 63)

# Loop over each index in the list
for idx in "${test_list[@]}"
do
    # Execute the command with the current index
    python3 run.py --world_idx $idx
done
