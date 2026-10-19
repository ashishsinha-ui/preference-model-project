#!/bin/bash
# One-shot overview of both queues: progress, current job, latest validation lines, finished results.
cd "$(dirname "$0")/.."
echo "################ $(date '+%F %T')"
for q in logs/week2_queue.out logs/week3_queue.out; do
    [ -f "$q" ] && { echo "== $q"; grep "===" "$q" | tail -n 4; }
done

current=$(ls -t logs/t*.log logs/mc_*.log logs/biasB_*.log 2>/dev/null | head -1)
echo -e "\n== running/latest: $current"
grep -E "^step|early stopping|Traceback|Error" "$current" | tail -n 4

echo -e "\n== finished runs (test-set accuracy)"
for f in logs/t*.log; do
    line=$(grep -E "^hh_test" "$f")
    [ -n "$line" ] && printf "%-36s %s\n" "$(basename "$f" .log)" "$(echo "$line" | cut -c1-110)"
done
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader | sed 's/^/\n== GPU: /'
