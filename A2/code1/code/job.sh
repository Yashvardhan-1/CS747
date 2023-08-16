#!/bin/sh
python3 encoder.py --states statesfilepath --q 0.25 > mdpfile && python3 planner.py --mdp mdpfile --policy ./data/cricket/original.txt  > rand_value_policy && python3 decoder.py --value-policy rand_value_policy  --states statesfilepath > rand_policyfile

python3 encoder.py --states statesfilepath --q 0.25 > mdpfile && python3 planner.py --mdp mdpfile > value_policy && python3 decoder.py --value-policy value_policy  --states statesfilepath > policyfile
