.PHONY: check test rtl-test synth synth-report

check: test rtl-test synth

test:
	python3 -m unittest discover -s . -p '*_test.py'

rtl-test:
	python3 -m fpga.verify

synth:
	yosys -Q -T -q -p 'read_verilog -sv fpga/rollout_engine.sv; hierarchy -top rollout_engine; synth -top rollout_engine -noabc; check -assert'

synth-report:
	yosys -Q -T -p 'read_verilog -sv fpga/rollout_engine.sv; hierarchy -top rollout_engine; synth -top rollout_engine -noabc; stat'
