#!/bin/bash
# Week 2 queue: runs one after another; each run logs to logs/<name>.log.
# A failed run is reported and the queue moves on.
cd "$(dirname "$0")/.."
export HF_HOME=$PWD/.hf_cache TOKENIZERS_PARALLELISM=false

run() {
    name=$1; shift
    echo "=== $(date '+%F %T') START $name"
    if "$@" > "logs/$name.log" 2>&1; then
        echo "=== $(date '+%F %T') DONE  $name"
    else
        echo "=== $(date '+%F %T') FAILED $name (see logs/$name.log)"
    fi
}

# missing T1 run
run t1_bilstm_cross_hard_hh python -m src.models.bilstm --dataset hh --label_type hard --mode cross

# T2 BERT-base
run t2_bert_hard_hh         python -m src.training.train --config configs/bert_base.yaml
run t2_bert_swapaug_hard_hh python -m src.training.train --config configs/bert_base.yaml swap_prob=0.5
run t2_bert_hard_shp        python -m src.training.train --config configs/bert_base.yaml dataset=shp
run t2_bert_soft_shp        python -m src.training.train --config configs/bert_base.yaml dataset=shp label_type=soft

# T3 DeBERTa-v3-base, main run
run t3_deberta_hard_hh      python -m src.training.train --config configs/deberta_base.yaml
