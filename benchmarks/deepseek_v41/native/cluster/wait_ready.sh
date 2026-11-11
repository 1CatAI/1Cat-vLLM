#!/bin/bash
# wait_ready.sh: block until the API on Dev-1:18000 answers, or a node logs an Error/Exception.
. $(dirname $0)/common.sh
while true; do
  ~/v41env/bin/python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:18000/v1/models', timeout=5)" 2> /dev/null && { echo READY; exit 0; }  # no curl on the nodes
  for h in $NODES; do
    t=$(ssh -o ConnectTimeout=10 $h 'grep -a -m1 -E "(Error|Exception):" ~/v41/logs/serve.log | grep -v Warning | head -1 | cut -c1-300')
    [ -n "$t" ] && { echo "FAIL $h: $t"; exit 1; }
  done
  sleep 20
done
