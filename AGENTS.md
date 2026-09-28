READ ~/bao-agent-scripts/AGENTS.md BEFORE ANYTHING (skip if missing).

This repository investigates deadline-aware planning with learned latent dynamics on FPGA. Start at `docs/README.md`. Keep measured results separate from hypotheses, and record the model, board, precision, host-transfer cost, and baseline settings with every performance claim. The current code gate is `python3 -m unittest discover -s worldbench -p '*_test.py'`; a smoke run is `python3 -m worldbench --episodes 4 --steps 6 --horizon 3`.
