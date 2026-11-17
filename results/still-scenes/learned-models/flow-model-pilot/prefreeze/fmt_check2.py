"""Pre-freeze format check, part 2: region extent against the 64 x 64 x 0.10 m raster for the long-structure rows of
fresh seed 100 (read-only fit seed). Same loading as fmt_check.py. No metric, no training."""
import sys
sys.path.insert(0, 'results/still-scenes/learned-models/flow-model-pilot/prefreeze')
import fmt_check as F  # noqa: E402

F.SCENES = [('eval', 'eval_06_waiting_line_1'), ('eval', 'eval_07_waiting_line_2'), ('eval', 'eval_13_waiting_line_3'),
            ('eval', 'eval_09_waiting_line_6'), ('eval', 'eval_14_audience_3'), ('examples', 'examples_07_audience_4'),
            ('eval', 'eval_10_conversation_group_2'), ('eval', 'eval_03_no_one_deep_place'), ('eval', 'eval_20_no_one_lift')]
F.part_a()
