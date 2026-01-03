#!/bin/bash

set -euo pipefail

KERNELS=$(python -c "
import sgemm
for name in sgemm.ALL_KERNELS:
  print(name)
")

OUTPUT_DIR="artifacts/profiles"
PROFILE_RANGE="sgemm_profile"


mkdir -p "$OUTPUT_DIR"

# ncu --set full -f -o "${OUTPUT_DIR}/${label}" \
#     --nvtx --nvtx-include "${PROFILE_RANGE}" \
#     python test.py

i=1
for kernel in $KERNELS; do
  label=$(printf "%02d_%s" "$i" "$kernel")
  ncu --set full -f -o "${OUTPUT_DIR}/${label}" \
    --nvtx --nvtx-include "${PROFILE_RANGE}" \
    python test.py --kernels "${kernel}" --skip-ref
  
  ncu --import "${OUTPUT_DIR}/${label}.ncu-rep" \
    --print-details all \
    --kernel-name-base demangled \
    --kernel-id ":::1" \
    > "${OUTPUT_DIR}/${label}.txt"
  
  ncu --import "${OUTPUT_DIR}/${label}.ncu-rep" \
    --page source \
    --print-source sass \
    --kernel-name-base demangled \
    --kernel-id ":::1" \
    > "${OUTPUT_DIR}/${label}.sass"
  
  ((i++))
done
