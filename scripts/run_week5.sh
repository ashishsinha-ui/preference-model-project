#!/bin/bash
# Week 5 queue (~2.5 h): reverse domain shift, UltraFeedback [AI-Label] ablation, LLM judge on the
# full HH test set, then significance tests and calibration. Small enough to share the GPU.
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

# existing HH models on the new eval sets (harmless split, UltraFeedback)
run eval_new_sets python -m src.evaluation.eval_checkpoint t2_bert_hard_hh_seed42 t3_deberta_hard_hh_seed42 --sets hhh_test uf_test

# reverse domain shift: train on harmless, test on helpful (hh_test) and harmless (hhh_test)
run t2_bert_hard_hhh python -m src.training.train --config configs/bert_base.yaml dataset=hhh

# [AI-Label] ablation: BERT trained on GPT-4-judged UltraFeedback, hard and soft
run t2_bert_hard_uf python -m src.training.train --config configs/bert_base.yaml dataset=uf
run t2_bert_soft_uf python -m src.training.train --config configs/bert_base.yaml dataset=uf label_type=soft

# LLM judge on the FULL HH test set (6194 pairs) so RQ1 has enough power; weights deleted afterwards
run t6_llama3_judge_full python -m src.models.llm_judge --model meta-llama/Meta-Llama-3-8B-Instruct --short_name llama3-8b --sets hh_val hh_test --max_n 0
hf cache rm -y model/meta-llama/Meta-Llama-3-8B-Instruct > /dev/null 2>&1

run calibrate_week5 python -m src.calibration.calibrate --all --no_track
run significance python -m src.evaluation.significance
echo "=== $(date '+%F %T') WEEK 5 QUEUE FINISHED"
