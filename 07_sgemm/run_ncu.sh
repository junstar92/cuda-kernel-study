#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

ARTIFACT_DIR="artifacts"
REPORT_FILE="${ARTIFACT_DIR}/sgemm_profile.ncu-rep"
PROFILE_RANGE="sgemm_profile"
PROFILE_CMD=(python test.py)

# Format: '<kernel id>|<output file name>'
# Fill this list directly in the script and the for-loop below will export each
# kernel's details into artifacts/<output file name>.
KERNEL_DETAIL_TARGETS=(
  '1|00_torch_matmul'
  '2|00_cutlass_128x128x8_64x32x8_2stage'
  '3|00_cutlass_128x128x8_32x64x8_2stage'
  '4|00_cutlass_universal_simt_128x256_8x4'
  '5|01_naive_row'
  '7|02_smem_tiling'
  '9|03_blocktiling_1d_64x64x8_8_pad4'
  '10|04_blocktiling_1d_64x64x8_16_pad4'
  '11|05_blocktiling_1d_128x128x8_64_pad4'
  '12|06_blocktiling_1d_128x128x16_64_pad4'
  '13|07_blocktiling_2d_128x128x8_8x8_pad4'
  '14|08_blocktiling_2d_128x128x16_8x8_pad4'
  '15|09_blocktiling_2d_vec4_128x128x8_8x8_pad4'
  '16|10_blocktiling_2d_vec4_128x128x16_8x8_pad4'
  '17|11_warptiling_v0_128x128x8_32x64x8'
  '18|12_warptiling_v0_128x128x8_64x32x8'
  '19|13_warptiling_v1_128x128x8_32x64x8'
  '20|14_warptiling_v1_128x128x8_64x32x8'
  '21|15_warptiling_v2_128x128x8_32x64x8'
  '22|16_warptiling_v2_128x128x8_64x32x8'
  '23|17_warptiling_v2_128x128x16_32x64x16'
  '24|18_warptiling_v3_128x128x8_32x64x8'
  '25|19_warptiling_v3_128x128x8_64x32x8'
  '26|20_warptiling_v3_128x128x16_32x64x16'
  '27|21_warptiling_v4_128x128x8_32x64x8'
  '28|22_warptiling_v4_128x128x8_64x32x8'
  '29|23_warptiling_v4_128x128x16_32x64x16'
  '30|24_warptiling_v5_128x128x8_32x64x8'
  '31|25_warptiling_v5_128x128x8_64x32x8'
  '32|26_warptiling_v5_128x128x16_32x64x16'
)

collect_report() {
  mkdir -p "${ARTIFACT_DIR}"

  ncu \
    --set full \
    --export "${REPORT_FILE}" \
    --force-overwrite \
    --nvtx \
    --nvtx-include "${PROFILE_RANGE}" \
    "${PROFILE_CMD[@]}"
}

write_kernel_details() {
  local kernel_id="$1"
  local output_name="$2"
  local output_path="${ARTIFACT_DIR}/${output_name}.txt"

  mkdir -p "$(dirname "${output_path}")"

  echo "extracting: ${kernel_id} -> ${output_path}"
  ncu \
    --import "${REPORT_FILE}" \
    --print-details all \
    --kernel-name-base demangled \
    --kernel-id ":::${kernel_id}" \
    > "${output_path}"
}

if ((${#KERNEL_DETAIL_TARGETS[@]} == 0)); then
  echo "KERNEL_DETAIL_TARGETS is empty"
  echo "Add '<kernel id>|<output file name>' entries to run_ncu.sh"
  return 1
fi

collect_report

for target in "${KERNEL_DETAIL_TARGETS[@]}"; do
  IFS='|' read -r kernel_id output_name <<< "${target}"
  write_kernel_details "${kernel_id}" "${output_name}"
done
