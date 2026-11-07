#!/bin/bash
# Item 12 step 1, the frozen block's render loop (I12_FROZEN.md, Run), unchanged.
cd .
I=results/still-scenes/fresh-scenes
ROWS="eval_00_no_one_place eval_01_no_one_place eval_02_no_one_place eval_03_no_one_deep_place eval_04_no_one_object eval_05_no_one_person eval_06_waiting_line_1 eval_07_waiting_line_2 eval_08_waiting_line_4 eval_09_waiting_line_6 eval_10_conversation_group_2 eval_11_conversation_group_3 eval_12_conversation_group_4 eval_13_waiting_line_3"
for N in ${SEEDS:-510 511 512 513 514}; do
  PYTHONPATH= .venv/bin/python -m sgi.synth.realistic.generate --variant real --seed $N --out $I/renders/seed$N --only $ROWS > $I/render_seed$N.log 2>&1; echo "$(date +%H:%M) seed $N exit $? pngs $(find $I/renders/seed$N -name '*.png' | wc -l)"
  grep -qi "out of memory" $I/render_seed$N.log && { echo "out of GPU memory: stopping"; exit 2; }
done
echo I12_RENDER_DONE
