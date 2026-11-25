#!/bin/bash
# Robust upload of cloud/work/upload/images -> cloud, in batches of 100 with retries (the host resets long streams).
HERE="$(cd "$(dirname "$0")" && pwd)"
. "$HERE/cloud.env" || { echo "cloud/cloud.env missing: copy cloud/cloud.env.example and fill it in"; exit 1; }
SSH="ssh -i $SSH_KEY -o BatchMode=yes -o ConnectTimeout=40 -o ServerAliveInterval=15 -p $SSH_PORT $SSH_USER@$SSH_HOST"
cd "$HERE/work/upload/images"
ls *.jpg | while read f; do echo "$f $(stat -c %s "$f")"; done | sort > /tmp/l_sizes.txt
for round in 1 2 3 4 5 6 7 8 9 10; do
  if ! $SSH 'cd /workspace/wadang/work/upload/images && find . -name "*.jpg" -printf "%f %s\n"' | sort > /tmp/r_sizes.txt; then sleep 20; continue; fi
  comm -23 /tmp/l_sizes.txt /tmp/r_sizes.txt | awk '{print $1}' > /tmp/missing.txt
  n=$(wc -l < /tmp/missing.txt); echo "round $round missing $n"
  [ "$n" -eq 0 ] && { echo SYNC_OK; exit 0; }
  split -l 100 /tmp/missing.txt /tmp/batch_
  for b in /tmp/batch_*; do timeout 600 bash -c "tar cf - -T $b | $SSH 'tar xf - -C /workspace/wadang/work/upload/images'" || sleep 10; rm -f $b; done
done
echo SYNC_INCOMPLETE
