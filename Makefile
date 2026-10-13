# refdiff: every number in the README comes from one of these targets.
# DeepPCB location: config/paths.yaml (or REFDIFF_DATA); big outputs: out_root (or REFDIFF_OUT).
SHELL := /bin/bash
.SHELLFLAGS := -o pipefail -c
PY    := ./.venv/bin/python
SEEDS ?= 0 1 2

.PHONY: diff-sweep shape yolo-ttd yolo-s ablations robustness pku-data pku space env setup-yolo data candidates train onnx registration detection yolo die2db failures compare table results gate demo test lint all

env:            ## venv with training, tests, layout (gdstk) and ONNX inference
	uv venv .venv --python 3.11
	uv pip install --python $(PY) -e '.[train,test,layout,infer]'

setup-yolo:     ## optional: the YOLOv8 baseline (ultralytics is AGPL-3.0; --no-deps keeps opencv-headless)
	uv pip install --python $(PY) -e '.[yolo]'
	uv pip install --python $(PY) --no-deps ultralytics==8.4.175

data:           ## clone DeepPCB (MIT, ~240 MB) to the configured path
	@d=$$($(PY) -c "from refdiff import data; print(data.data_root().parent)"); \
	  [ -d "$$d/PCBData" ] || git clone --depth 1 https://github.com/tangsanli5201/DeepPCB.git "$$d"

candidates:     ## diff stage over trainval (native, warp) and test (all 4 scenarios), cached as .npz
	$(PY) scripts/build_candidates.py --split trainval --scenarios native,warp
	$(PY) scripts/build_candidates.py --split test --scenarios native,native_noreg,warp,warp_noreg

train:          ## candidate classifier, one model per seed (MPS)
	for s in $(SEEDS); do $(PY) -u scripts/train_classifier.py --seed $$s | tee results/train_classifier_s$$s.log; done

onnx:           ## export seed 0 to models/candnet.onnx (committed)
	$(PY) scripts/export_onnx.py

registration:   ## results/registration.json
	$(PY) scripts/eval_registration.py | tee results/registration.log

detection:      ## results/detection.json (+ bootstrap CIs, seed spread)
	$(PY) scripts/eval_detection.py | tee results/detection.log

yolo:           ## template-blind YOLOv8n baseline -> results/yolo_yolov8n.json (~1 h on an M3 Max)
	cd weights && ../$(PY) -u ../scripts/yolo_baseline.py --model yolov8n.pt

yolo-ttd:       ## YOLOv8n given the reference: registered test / template / signed diff -> results/yolo_yolov8n_ttd.json
	cd weights && ../$(PY) -u ../scripts/yolo_baseline.py --model yolov8n.pt --input ttd

yolo-s:         ## a bigger template-blind baseline -> results/yolo_yolov8s.json
	cd weights && ../$(PY) -u ../scripts/yolo_baseline.py --model yolov8s.pt

shape:          ## refdiff candidates classified from geometry alone (no CNN) -> results/shape_baseline.json
	$(PY) scripts/shape_baseline.py

ablations:      ## classifier input / training-data ablations (trains 5 variants) -> results/ablations.json
	$(PY) -u scripts/ablations.py | tee results/ablations.log

diff-sweep:     ## XOR tolerance x min blob area -> results/diff_sweep.json
	$(PY) scripts/diff_sweep.py

robustness:     ## mAP vs rotation / blur / noise -> results/robustness.json, docs/robustness.png
	$(PY) -u scripts/robustness.py | tee results/robustness.log

pku-data:       ## PKU synthetic PCB defects from Kaggle (free account; token in ~/.kaggle/access_token)
	@d=$$($(PY) -c "from refdiff import pku; print(pku.root().parent)"); mkdir -p "$$d" && cd "$$d" && \
	  curl -sSL -H "Authorization: Bearer $$(cat ~/.kaggle/access_token)" -o pcb-defects.zip \
	  https://www.kaggle.com/api/v1/datasets/download/akhatova/pcb-defects && unzip -q -o pcb-defects.zip

pku:            ## cross-dataset test on PKU, no retraining -> results/pku.json
	$(PY) -u scripts/pku_transfer.py | tee results/pku.log

die2db:         ## GDSII die-to-database stretch -> results/die2db.json
	$(PY) scripts/die2db.py

failures:       ## every error tagged with a cause -> results/failures.json, docs/failures/*.jpg
	$(PY) scripts/failure_analysis.py

compare:        ## paired bootstraps: refdiff vs YOLO, registration on vs off -> results/compare.json
	$(PY) scripts/compare.py

table:          ## regenerate the README results tables from results/*.json
	$(PY) scripts/results_table.py --write

results: candidates train onnx registration detection die2db compare failures table

gate:           ## the CI quality gate (DeepPCB + committed ONNX model; no torch)
	$(PY) scripts/gate.py

demo:           ## one test pair through the pipeline -> docs/demo.png
	$(PY) scripts/demo.py

test:
	$(PY) -m pytest -q

lint:
	$(PY) -m ruff check refdiff scripts tests

all: lint test gate

space:          ## assemble build/space/ for a Hugging Face Space (app + package + ONNX model + examples); does not upload
	rm -rf build/space && mkdir -p build/space/models
	cp space/app.py space/README.md space/requirements.txt build/space/
	cp -R space/examples build/space/
	cp -R refdiff build/space/ && rm -rf build/space/refdiff/__pycache__
	cp models/candnet.onnx build/space/models/
	@echo "build/space ready: run it with  cd build/space && python app.py"
