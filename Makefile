.PHONY: check model-check rtl-synth xcup-report

check: rtl-synth xcup-report
	python3 -m unittest discover -s worldmodel -p '*_test.py'
	python3 -m worldmodel.verify_rtl
	python3 -m worldmodel.verify_candidate_rtl

rtl-synth:
	yosys -Q -T -q -p 'read_verilog -sv rtl/qkv_tile.sv; hierarchy -top qkv_tile; synth -top qkv_tile -noabc; check -assert'
	yosys -Q -T -q -p 'read_verilog -sv rtl/qkv_candidate_tile.sv; hierarchy -top qkv_candidate_tile; synth -top qkv_candidate_tile -noabc; check -assert'

xcup-report:
	python3 -m worldmodel.resource_report

model-check:
	.venv/bin/python -m worldmodel.bench --device cpu --frames 2
	.venv/bin/python -m worldmodel.verify_rtl --checkpoint .model-cache/models--facebook--jepa-wms/snapshots/bb2d9cf0ee9060f83103b134d7c52e82bf7e2a47/jepa_wm_pusht.pth.tar
	.venv/bin/python -m worldmodel.candidate_bench --counts 1 2 --repeats 1 --warmup 0
	.venv/bin/python -m worldmodel.verify_candidate_rtl --checkpoint .model-cache/models--facebook--jepa-wms/snapshots/bb2d9cf0ee9060f83103b134d7c52e82bf7e2a47/jepa_wm_pusht.pth.tar
	.venv/bin/python -m unittest worldmodel.activation_probe_test worldmodel.quant_test
	.venv/bin/python -m worldmodel.activation_probe --tokens 0
