#!/usr/bin/env python3
from common import load_cli_config
from fibre_sim.pipeline import generate_motion_step

generate_motion_step(load_cli_config("Step 01: generate object trajectory"))

