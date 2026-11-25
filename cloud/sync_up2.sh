#!/bin/bash
# Upload working copies to the NEW machine in reverse order, batches of 100; skips files already present with same size.
HERE="$(cd "$(dirname "$0")" && pwd)"
. "$HERE/cloud.env" || { echo "cloud/cloud.env missing: copy cloud/cloud.env.example and fill it in"; exit 1; }
SSH="ssh -i $SSH_KEY -o BatchMode=yes -o ConnectTimeout=40 -o ServerAliveInterval=15 -p $SSH_PORT $SSH_USER@$SSH_HOST"
cd "$HERE/work/upload/images"
ls *.jpg | while read f; do echo "$f $(stat -c %s "$f")"; done | sort > /tmp/l2_sizes.txt
for round in 1 2 3 4 5 6 7 8; do
  $SSH 'mkdir -p /workspace/wadang/work/upload/images && cd /workspace/wadang/work/upload/images && find . -name "*.jpg" -printf "%f %s\n"' | sort > /tmp/r2_sizes.txt || { sleep 15; continue; }
  comm -23 /tmp/l2_sizes.txt /tmp/r2_sizes.txt | awk '{print $1}' | sort -r > /tmp/missing2.txt
  n=$(wc -l < /tmp/missing2.txt); echo "round $round missing $n"
  [ "$n" -eq 0 ] && { echo SYNC2_OK; exit 0; }
  head -400 /tmp/missing2.txt | split -l 100 - /tmp/b2_
  for b in /tmp/b2_*; do timeout 600 bash -c "tar cf - -T $b | $SSH 'tar xf - -C /workspace/wadang/work/upload/images'" || sleep 10; rm -f $b; done
done
echo SYNC2_INCOMPLETE
