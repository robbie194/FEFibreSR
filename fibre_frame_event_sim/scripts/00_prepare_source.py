#!/usr/bin/env python3
from common import load_cli_config
from fibre_sim.pipeline import prepare_source_step

prepare_source_step(load_cli_config("Step 00: crop and calibrate source image"))

