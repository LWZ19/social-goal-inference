"""Run sgi.smoke with the E3 config (the seven older models added) in place of configs/default.yaml.
Usage: smoke_with_config.py CONFIG <sgi.smoke arguments>"""
import sys
from pathlib import Path
import sgi.config
cfg = Path(sys.argv[1]).resolve()
sgi.config.load_config.__defaults__ = (cfg, None)
import sgi.smoke
sys.exit(sgi.smoke.main(sys.argv[2:]))
