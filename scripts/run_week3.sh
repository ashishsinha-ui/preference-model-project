#!/bin/bash
# Week 3 queue: waits for the Week 2 queue to finish, then runs Test B, MC dropout,
# the remaining DeBERTa-base runs and calibration. Each step logs to logs/<name>.log.
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

echo "=== $(date '+%F %T') waiting for run_week2.sh to finish"
while pgrep -f run_week2.sh > /dev/null; do sleep 60; done

# Test B + MC dropout on the Week 2 models
run biasB_week2          python -m src.evaluation.bias_tests evaluate t2_bert_hard_hh_seed42 t2_bert-swapaug_hard_hh_seed42 t3_deberta_hard_hh_seed42
run mc_t2_bert_hard_hh   python -m src.calibration.mc_dropout t2_bert_hard_hh_seed42
run mc_t2_bert_soft_shp  python -m src.calibration.mc_dropout t2_bert_soft_shp_seed42

# T3 DeBERTa-v3-base: swap augmentation (RQ2) and SHP hard vs soft (RQ3)
run t3_deberta_swapaug_hard_hh python -m src.training.train --config configs/deberta_base.yaml swap_prob=0.5
run t3_deberta_hard_shp        python -m src.training.train --config configs/deberta_base.yaml dataset=shp
run t3_deberta_soft_shp        python -m src.training.train --config configs/deberta_base.yaml dataset=shp label_type=soft

run biasB_t3                python -m src.evaluation.bias_tests evaluate t3_deberta-swapaug_hard_hh_seed42
run mc_t3_deberta_hard_hh   python -m src.calibration.mc_dropout t3_deberta_hard_hh_seed42
run mc_t3_deberta_soft_shp  python -m src.calibration.mc_dropout t3_deberta_soft_shp_seed42

# calibration on every run with saved predictions (CPU)
run calibrate_all python -m src.calibration.calibrate --all --no_track
