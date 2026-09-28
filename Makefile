.PHONY: check model-check rtl-synth

check: rtl-synth
	python3 -m unittest discover -s worldmodel -p '*_test.py'
	python3 -m worldmodel.verify_rtl

rtl-synth:
	yosys -Q -T -q -p 'read_verilog -sv rtl/qkv_tile.sv; hierarchy -top qkv_tile; synth -top qkv_tile -noabc; check -assert'

model-check:
	.venv/bin/python -m worldmodel.bench --device cpu --frames 2
	.venv/bin/python -m worldmodel.verify_rtl --checkpoint .model-cache/models--facebook--jepa-wms/snapshots/bb2d9cf0ee9060f83103b134d7c52e82bf7e2a47/jepa_wm_pusht.pth.tar
