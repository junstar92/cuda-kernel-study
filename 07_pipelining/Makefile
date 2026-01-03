# ─────────────────────────────────────────────
#  Makefile for CUDA Pipelining Study
# ─────────────────────────────────────────────
NVCC      	:= nvcc
ARCH      	:= sm_80
STAGES    	?= 2
NVCCFLAGS 	:= -O3 -arch=$(ARCH) -std=c++17 --expt-relaxed-constexpr
OUTPUT_DIR 	?= ./artifacts

TARGETS := $(OUTPUT_DIR)/baseline $(OUTPUT_DIR)/ping_pong $(OUTPUT_DIR)/cp_async

.PHONY: all clean profile run

all: $(TARGETS) | $(OUTPUT_DIR)

$(OUTPUT_DIR)/baseline: baseline_no_pipeline.cu | $(OUTPUT_DIR)
	$(NVCC) $(NVCCFLAGS) -o $@ $<

$(OUTPUT_DIR)/ping_pong: ping_pong.cu | $(OUTPUT_DIR)
	$(NVCC) $(NVCCFLAGS) -o $@ $<

$(OUTPUT_DIR)/cp_async: cp_async.cu | $(OUTPUT_DIR)
	$(NVCC) $(NVCCFLAGS) -DSTAGES=$(STAGES) -o $@ $<


$(OUTPUT_DIR):
	mkdir -p $@


# ─────────────────────────────────────────────
#  Generate PTX
# ─────────────────────────────────────────────
ptx: all | $(OUTPUT_DIR)
	cuobjdump -ptx $(OUTPUT_DIR)/baseline > $(OUTPUT_DIR)/baseline.ptx
	cuobjdump -ptx $(OUTPUT_DIR)/ping_pong > $(OUTPUT_DIR)/ping_pong.ptx
	cuobjdump -ptx $(OUTPUT_DIR)/cp_async > $(OUTPUT_DIR)/cp_async.ptx


# ─────────────────────────────────────────────
#  Generate SASS
# ─────────────────────────────────────────────
sass: all | $(OUTPUT_DIR)
	cuobjdump -sass $(OUTPUT_DIR)/baseline > $(OUTPUT_DIR)/baseline.sass
	cuobjdump -sass $(OUTPUT_DIR)/ping_pong > $(OUTPUT_DIR)/ping_pong.sass
	cuobjdump -sass $(OUTPUT_DIR)/cp_async > $(OUTPUT_DIR)/cp_async.sass


# ─────────────────────────────────────────────
#  Run all benchmarks
# ─────────────────────────────────────────────
run: all
	@echo "════════════════════════════════════════"
	@echo " Running Baseline"
	@echo "════════════════════════════════════════"
	$(OUTPUT_DIR)/baseline
	@echo ""
	@echo "════════════════════════════════════════"
	@echo " Running Ping-Pong"
	@echo "════════════════════════════════════════"
	$(OUTPUT_DIR)/ping_pong
	@echo ""
	@echo "════════════════════════════════════════"
	@echo " Running cp.async " $(STAGES) "-Stage"
	@echo "════════════════════════════════════════"
	$(OUTPUT_DIR)/cp_async

# ─────────────────────────────────────────────
#  NCU profiling
# ─────────────────────────────────────────────
profile: all | $(OUTPUT_DIR)
	ncu --set full -f -o $(OUTPUT_DIR)/baseline_report --kernel-name-base demangled --kernel-id :::1 $(OUTPUT_DIR)/baseline
	ncu --set full -f -o $(OUTPUT_DIR)/ping_pong_report --kernel-name-base demangled --kernel-id :::1 $(OUTPUT_DIR)/ping_pong
	ncu --set full -f -o $(OUTPUT_DIR)/cp_async_report --kernel-name-base demangled --kernel-id :::1 $(OUTPUT_DIR)/cp_async
	@echo ""
	ncu --import $(OUTPUT_DIR)/baseline_report.ncu-rep --print-details all > $(OUTPUT_DIR)/baseline_report_details.txt
	ncu --import $(OUTPUT_DIR)/ping_pong_report.ncu-rep --print-details all > $(OUTPUT_DIR)/ping_pong_report_details.txt
	ncu --import $(OUTPUT_DIR)/cp_async_report.ncu-rep --print-details all > $(OUTPUT_DIR)/cp_async_report_details.txt
	@echo ""
	

clean:
	rm -rf $(TARGETS) $(OUTPUT_DIR)