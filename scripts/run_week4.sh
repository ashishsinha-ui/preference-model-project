#!/bin/bash
# Week 4 queue: waits for the DeBERTa queue to finish, then runs
#   T5 bi-encoder, no-prompt ablation, deep ensembles (BERT + DeBERTa), Test B on the new models,
#   T6 LLM judge (LLaMA-3-8B, falls back to Qwen2.5-7B if LLaMA fails), calibration.
# Each step logs to logs/<name>.log; a failed step is reported and the queue moves on.
cd "$(dirname "$0")/.."
export HF_HOME=$PWD/.hf_cache TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1

run() {
    name=$1; shift
    echo "=== $(date '+%F %T') START $name"
    if "$@" > "logs/$name.log" 2>&1; then
        echo "=== $(date '+%F %T') DONE  $name"
    else
        echo "=== $(date '+%F %T') FAILED $name (see logs/$name.log)"
    fi
}

echo "=== $(date '+%F %T') waiting for rerun_deberta.sh to finish"
while pgrep -f rerun_deberta.sh > /dev/null; do sleep 60; done

# T5 bi-encoder: position-invariant by construction, no cross-attention between A and B
run t5_bert-bi_hard_hh  python -m src.training.train --config configs/bert_bi.yaml
run t5_bert-bi_soft_shp python -m src.training.train --config configs/bert_bi.yaml dataset=shp label_type=soft

# Ablation: response-only model (no prompt context)
run t2_bert-noprompt_hard_hh python -m src.training.train --config configs/bert_base.yaml max_prompt=0

# Deep ensemble, BERT: 2 more seeds + the existing seed 42
run t2_bert_hard_hh_seed1 python -m src.training.train --config configs/bert_base.yaml seed=1
run t2_bert_hard_hh_seed2 python -m src.training.train --config configs/bert_base.yaml seed=2
run ens_t2_bert python -m src.calibration.ensemble t2_bert_hard_hh_seed42 t2_bert_hard_hh_seed1 t2_bert_hard_hh_seed2

# Test B on the new architectures
run biasB_week4 python -m src.evaluation.bias_tests evaluate t5_bert-bi_hard_hh_seed42 t2_bert-noprompt_hard_hh_seed42

# T6 LLM judge (1000 pairs per eval set, both orders); delete the 16 GB of weights afterwards
run t6_llama3_judge python -m src.models.llm_judge --model meta-llama/Meta-Llama-3-8B-Instruct --short_name llama3-8b
hf cache rm -y model/meta-llama/Meta-Llama-3-8B-Instruct > /dev/null 2>&1
if [ ! -f results/t6_llama3-8b-3shot_judge_hh_seed42/metrics.json ]; then
    run t6_qwen_judge python -m src.models.llm_judge --model Qwen/Qwen2.5-7B-Instruct --short_name qwen2.5-7b
    hf cache rm -y model/Qwen/Qwen2.5-7B-Instruct > /dev/null 2>&1
fi

# Deep ensemble, DeBERTa (plan: 3 seeds x DeBERTa-v3-base; fp32, ~1.5 h each)
run t3_deberta_hard_hh_seed1 python -m src.training.train --config configs/deberta_base.yaml seed=1
run t3_deberta_hard_hh_seed2 python -m src.training.train --config configs/deberta_base.yaml seed=2
run ens_t3_deberta python -m src.calibration.ensemble t3_deberta_hard_hh_seed42 t3_deberta_hard_hh_seed1 t3_deberta_hard_hh_seed2

run calibrate_all_week4 python -m src.calibration.calibrate --all --no_track
echo "=== $(date '+%F %T') WEEK 4 QUEUE FINISHED"
