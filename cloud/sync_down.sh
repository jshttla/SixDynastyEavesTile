#!/bin/bash
# Pull GPU outputs back: work/out/{std,overlay} in batches of 300 + small files. Retries; verifies counts.
HERE="$(cd "$(dirname "$0")" && pwd)"
. "$HERE/cloud.env" || { echo "cloud/cloud.env missing: copy cloud/cloud.env.example and fill it in"; exit 1; }
SSH="ssh -i $SSH_KEY -o BatchMode=yes -o ConnectTimeout=40 -o ServerAliveInterval=15 -p $SSH_PORT $SSH_USER@$SSH_HOST"
L="$HERE/work/out"
mkdir -p $L/std $L/overlay
for f in seg_index.csv rim.csv dinov2_emb.npz tilt_validation.json pip_freeze.txt gpu.txt started.txt finished.txt; do
  for t in 1 2 3 4 5; do scp -i $SSH_KEY -P $SSH_PORT -o ConnectTimeout=40 -q $SSH_USER@$SSH_HOST:/workspace/wadang/work/out/$f $L/ && break; sleep 10; done
done
for d in std overlay; do
  for round in 1 2 3 4 5 6; do
    $SSH "cd /workspace/wadang/work/out/$d && find . -type f -printf \"%f %s\n\"" | sort > /tmp/r_$d.txt || { sleep 15; continue; }
    (cd $L/$d && for f in *; do [ -f "$f" ] && echo "$f $(stat -c %s "$f")"; done) | sort > /tmp/l_$d.txt
    comm -23 /tmp/r_$d.txt /tmp/l_$d.txt | awk '{print $1}' > /tmp/need_$d.txt
    n=$(wc -l < /tmp/need_$d.txt); echo "$d round $round need $n"
    [ "$n" -eq 0 ] && break
    split -l 300 /tmp/need_$d.txt /tmp/dl_${d}_
    for b in /tmp/dl_${d}_*; do
      timeout 900 bash -c "cat $b | $SSH 'cd /workspace/wadang/work/out/$d && tar cf - -T -' | tar xf - -C $L/$d" || sleep 10
      rm -f $b
    done
  done
done
echo "std $(ls $L/std | wc -l) overlay $(ls $L/overlay | wc -l)"
echo SYNC_DOWN_DONE
