#!/bin/bash

read -p "Enter the string to replace <YOUR_VGGS_PATH> with: " VGGS_PATH

for f in *template*.json; do
    out="${f/_template/}"
    sed "s|<YOUR_VGGS_PATH>|${VGGS_PATH}|g" "$f" > "$out"
    echo "Generated $out"
done
