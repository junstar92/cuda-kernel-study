#!/bin/bash

ncu --set full -o artifacts/sgemm_profile.ncu-rep -f --nvtx --nvtx-include sgemm_profile python test.py
ncu --print-details all -i artifacts/sgemm_profile.ncu-rep > artifacts/sgemm_profile_details.txt