#!/usr/bin/env python3
from common import load_cli_config
from fibre_sim.pipeline import validate_outputs_step

validate_outputs_step(load_cli_config("Step 07: validate all simulation outputs"))

