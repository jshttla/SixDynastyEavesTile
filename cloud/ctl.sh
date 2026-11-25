#!/bin/bash
# Process control without pkill -f (pattern matching killed the controlling ssh session twice).
# usage: bash ctl.sh start NAME CMD... | stop NAME | status NAME
cd /workspace/wadang
export HF_HOME=/workspace/wadang/models/hf HF_HUB_OFFLINE=1
mkdir -p run logs
case "$1" in
  start) name=$2; shift 2; nohup "$@" > logs/$name.log 2>&1 & echo $! > run/$name.pid; echo "started $name pid $(cat run/$name.pid)";;
  stop)  [ -f run/$2.pid ] && kill $(cat run/$2.pid) 2>/dev/null; rm -f run/$2.pid; echo "stopped $2";;
  status) [ -f run/$2.pid ] && kill -0 $(cat run/$2.pid) 2>/dev/null && echo "$2 running" || echo "$2 not running";;
esac
