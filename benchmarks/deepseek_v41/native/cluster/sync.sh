#!/bin/bash
# sync.sh [PATH...]: copy files from Dev-1 ~/v41/src to Dev-2..4 (default: every file
# git reports as changed vs HEAD, plus the launcher/config in ~/v41).
. $(dirname $0)/common.sh
cd $SRC || exit 1
FILES="$@"
[ -z "$FILES" ] && FILES=$(git status --porcelain | awk '{print $NF}')
for h in 10.56.0.12 10.56.0.13 10.56.0.14; do
  ( [ -n "$FILES" ] && rsync -aR $FILES $h:v41/src/
    rsync -a ~/v41/run_pp4.sh ~/v41/run_pp4_nsys.sh ~/v41/serve-native-pp4.yaml $h:v41/
    echo "synced $h: $(echo $FILES | wc -w) files" ) &
done
wait
