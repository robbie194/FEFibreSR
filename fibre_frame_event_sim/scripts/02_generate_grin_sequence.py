#!/usr/bin/env python3
from common import load_cli_config
from fibre_sim.pipeline import generate_grin_step

generate_grin_step(load_cli_config("Step 02: simulate moving image through GRIN relay"))

