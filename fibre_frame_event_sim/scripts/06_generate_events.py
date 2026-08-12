#!/usr/bin/env python3
from common import load_cli_config
from fibre_sim.pipeline import generate_events_step

generate_events_step(load_cli_config("Step 06: generate v2e event stream"))

