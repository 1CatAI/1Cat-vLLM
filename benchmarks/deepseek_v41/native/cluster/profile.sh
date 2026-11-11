#!/bin/bash
# profile.sh ["ENV=VAL ..."]: restart under nsys, capture a 200-token decode + one 14.5K
# prefill (profile_both.py), export sqlite on every node, print the per-stage decode ledger
# (decode_ledger.py) and prefill ledger. Reports land in ~/v41/prof/stage{0..3}.sqlite per node.
. $(dirname $0)/common.sh
$(dirname $0)/stop.sh > /dev/null
$(dirname $0)/start.sh run_pp4_nsys.sh "$1" > /dev/null
$(dirname $0)/wait_ready.sh || exit 1
cd ~/v41 && ~/v41env/bin/python profile_both.py
for r in 0 1 2 3; do
  ssh $(rank_ip $r) "cd ~/v41/prof; for i in \$(seq 90); do [ -s stage$r.nsys-rep ] && break; sleep 10; done; sleep 20; nsys export --type sqlite -f true -o stage$r.sqlite stage$r.nsys-rep > export.log 2>&1; ~/v41env/bin/python ~/v41/decode_ledger.py stage$r.sqlite; ~/v41env/bin/python ~/v41/prefill_ledger.py stage$r.sqlite" > ~/v41/logs/prof_stage$r.log 2>&1 &
done
wait
cat ~/v41/logs/prof_stage{0,1,2,3}.log
