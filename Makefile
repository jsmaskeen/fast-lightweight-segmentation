ifeq ($(OS),Windows_NT)
    PYTHON ?= .venv/Scripts/python.exe
else
    PYTHON ?= .venv/bin/python
endif

.PHONY: all data train distil distil-quant distil-bench export quant eval bench extA extB studies report visuals buvfm test clean webviewer serve

all: data train distil distil-quant distil-bench eval export quant extA bench extB studies report visuals

data:
	$(PYTHON) -m src.data.download_busi
	$(PYTHON) -m src.data.merge_masks

train:
	$(PYTHON) -m src.train.train --model r34 --epoch 100
	$(PYTHON) -m src.train.train --model mbv3s --epoch 100

distil:
	$(PYTHON) -m src.distil.distil --teacher-model r34 --teacher-ckpt models/fp32/r34_best.pt --student-model mbv3s --epochs 100

distil-quant:
	$(PYTHON) -m src.distil.quantize --model-name mbv3s --method both

distil-bench:
	$(PYTHON) -m src.distil.bench

eval:
	$(PYTHON) -m src.metrics.eval_baseline

export:
	$(PYTHON) -m src.export.to_onnx --model r34
	$(PYTHON) -m src.export.to_onnx --model mbv3s

quant:
	$(PYTHON) -m src.quant.quantize --model r34
	$(PYTHON) -m src.quant.quantize --model mbv3s
	$(PYTHON) -m src.quant.audit

bench:
	$(PYTHON) -m src.bench.harness

extA:
	$(PYTHON) -m src.quant.sensitivity
	$(PYTHON) -m src.quant.mixed
	$(PYTHON) -m src.quant.demo_page

extB:
	$(PYTHON) -m src.cine.run_cine

studies:
	$(PYTHON) -m src.metrics.stratify
	$(PYTHON) -m src.bench.op_profile
	$(PYTHON) -m src.quant.calib_study

report:
	$(PYTHON) -m src.report.make_figures
	$(PYTHON) -m src.report.make_tables

visuals:
	$(PYTHON) -m src.bench.literature_compare
	$(PYTHON) -m src.report.make_benchmark_charts
	$(PYTHON) -m src.report.make_overlay_plot
	$(PYTHON) -m src.report.make_overlay_normal
	$(PYTHON) -m src.cine.visual_cine
	$(PYTHON) -m src.cine.multi_demo_generator
	$(PYTHON) -m src.cine.buv_multi_tracker
	$(PYTHON) -m src.report.make_tables

buvfm:
	$(PYTHON) -m src.metrics.buvfm_eval
	$(PYTHON) -m src.cine.buv_multi_tracker

webviewer:
	$(PYTHON) -m server.app --host 0.0.0.0 --port 8000

serve: webviewer

test:
	$(PYTHON) -m pytest tests/ -v

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name "*.pyo" -delete 2>/dev/null || true
