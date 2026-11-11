#!/bin/bash
# wait_ready.sh: block until the API on Dev-1 answers or any node logs a Traceback.
while true; do
  ssh -o ControlPath=none -o ConnectTimeout=15 c4140-1 'curl -s -m 5 http://127.0.0.1:18000/v1/models >/dev/null 2>&1 || python3 -c "import urllib.request;urllib.request.urlopen(\"http://127.0.0.1:18000/v1/models\",timeout=5)" 2>/dev/null' && { echo READY; exit 0; }
  for n in 1 2 3 4; do
    t=$(ssh -o ControlPath=none -o ConnectTimeout=15 c4140-$n 'grep -a -m1 -A0 -E "^.*(Error|Exception):" ~/v41/logs/serve.log | grep -vE "Warning" | head -1 | cut -c1-300' 2>/dev/null)
    [ -n "$t" ] && { echo "FAIL n$n: $t"; exit 1; }
  done
  sleep 30
done
