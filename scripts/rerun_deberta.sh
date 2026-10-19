#!/bin/bash
# Re-run every DeBERTa job that failed on 2026-10-13 (NaN under bf16 autocast).
# 2026-10-19: swap-aug and hard-SHP finished; DataLoaders now use num_workers=0
# (soft-SHP deadlocked in a forked worker for 5 days).
# DeBERTa now trains in fp32 with batch 16 x grad_accum 2 (configs/deberta_base.yaml).
# Each step logs to logs/<name>.log; a failed step is reported and the queue moves on.
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

# T3 DeBERTa-v3-base: main run, swap augmentation (RQ2), SHP hard vs soft (RQ3)
run t3_deberta_hard_hh         python -m src.training.train --config configs/deberta_base.yaml
# done 2026-10-14: run t3_deberta_swapaug_hard_hh python -m src.training.train --config configs/deberta_base.yaml swap_prob=0.5
# done 2026-10-14: run t3_deberta_hard_shp        python -m src.training.train --config configs/deberta_base.yaml dataset=shp
run t3_deberta_soft_shp        python -m src.training.train --config configs/deberta_base.yaml dataset=shp label_type=soft

# Test B + MC dropout on the DeBERTa models
run biasB_t3               python -m src.evaluation.bias_tests evaluate t3_deberta_hard_hh_seed42 t3_deberta-swapaug_hard_hh_seed42
run mc_t3_deberta_hard_hh  python -m src.calibration.mc_dropout t3_deberta_hard_hh_seed42
run mc_t3_deberta_soft_shp python -m src.calibration.mc_dropout t3_deberta_soft_shp_seed42

# calibration on every run with saved predictions (CPU)
run calibrate_all python -m src.calibration.calibrate --all --no_track
