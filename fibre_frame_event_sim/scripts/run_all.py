#!/usr/bin/env python3
from common import load_cli_config
from fibre_sim.pipeline import run_all

run_all(load_cli_config("Run the complete phase-one Frame/Event simulation"))

