#!/bin/bash

#SBATCH --job-name=prompt_steal
#SBATCH --nodes=1
#SBATCH --partition=killable
#SBATCH --nodelist=n-805
#SBATCH --gpus=1
#SBATCH --ntasks-per-node=1
#SBATCH --array=3
#SBATCH --output=logs/run_%A_%a.out
#SBATCH --error=logs/run_%A_%a.err

# Create logs directory
mkdir -p logs

# Define datasets for each array task
DATASETS=(sst2 qqp mnli qnli)
TASK=${DATASETS[$SLURM_ARRAY_TASK_ID]}

echo "=========================================="
echo "SLURM Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Processing dataset: $TASK"
echo "=========================================="

# Experimental parameters
SEEDS=(42 43 44)
BUDGETS=(200 500 1000 5000)
EPOCHS=20
BATCH_SIZE_TRAIN=256
BATCH_SIZE_EVAL=64
LR=1e-2
EPSILON=8.0

# Directories
CKPT_DIR="checkpoints/${TASK}"
RESULTS_DIR="results/${TASK}"
mkdir -p ${CKPT_DIR}
mkdir -p ${RESULTS_DIR}

echo ""
echo "Configuration:"
echo "  Seeds: ${SEEDS[@]}"
echo "  Budgets: ${BUDGETS[@]}"
echo "  Epochs: ${EPOCHS}"
echo "  Checkpoint dir: ${CKPT_DIR}"
echo "  Results dir: ${RESULTS_DIR}"
echo ""

# Loop over seeds
for SEED in "${SEEDS[@]}"; do
    echo "================================================"
    echo "SEED ${SEED} - Training Victim Models"
    echo "================================================"
    
    # Train victim (no DP)
    VICTIM_CKPT="${CKPT_DIR}/victim_${TASK}_P20_seed${SEED}.pt"
    if [ ! -f "${VICTIM_CKPT}" ]; then
        echo "[${TASK}] Training victim (no DP) with seed ${SEED}..."
        python train_victim.py \
            --task ${TASK} \
            --epochs 3 \
            --save_dir ${CKPT_DIR} \
            --lr ${LR} \
            --batch_size ${BATCH_SIZE_TRAIN} \
            --seed ${SEED} \
            --prompt_len 20
        echo "✓ Victim (no DP) saved to ${VICTIM_CKPT}"
    else
        echo "✓ Victim (no DP) already exists: ${VICTIM_CKPT}"
    fi
    
    # Train victim (DP)
    VICTIM_DP_CKPT="${CKPT_DIR}/victim_dp_${TASK}_eps${EPSILON}_P20_seed${SEED}.pt"
    if [ ! -f "${VICTIM_DP_CKPT}" ]; then
        echo "[${TASK}] Training victim (DP) with seed ${SEED}..."
        python train_victim_dp.py \
            --task ${TASK} \
            --epochs 3 \
            --save_dir ${CKPT_DIR} \
            --lr ${LR} \
            --batch_size ${BATCH_SIZE_TRAIN} \
            --seed ${SEED} \
            --epsilon ${EPSILON} \
            --prompt_len 20
        echo "✓ Victim (DP) saved to ${VICTIM_DP_CKPT}"
    else
        echo "✓ Victim (DP) already exists: ${VICTIM_DP_CKPT}"
    fi
    
    # Loop over budgets
    for BUDGET in "${BUDGETS[@]}"; do
        echo ""
        echo "================================================"
        echo "SEED ${SEED} | BUDGET ${BUDGET} - Stealing"
        echo "================================================"
        
        # --- SOFT ORACLE (NO DP) ---
        STOLEN_SOFT="${CKPT_DIR}/stolen_${TASK}_nodp_budget${BUDGET}_seed${SEED}_soft.pt"
        if [ ! -f "${STOLEN_SOFT}" ]; then
            echo "[${TASK}] Stealing (soft, no DP) | budget=${BUDGET}, seed=${SEED}..."
            python steal_prompt.py \
                --task ${TASK} \
                --victim_ckpt ${VICTIM_CKPT} \
                --out_ckpt ${STOLEN_SOFT} \
                --budget ${BUDGET} \
                --epochs ${EPOCHS} \
                --batch_size ${BATCH_SIZE_TRAIN} \
                --lr ${LR} \
                --seed ${SEED} \
                --oracle probs
            echo "✓ Stolen (soft, no DP) saved"
        else
            echo "✓ Stolen (soft, no DP) already exists"
        fi
        
        # --- HARD ORACLE (NO DP) ---
        STOLEN_HARD="${CKPT_DIR}/stolen_${TASK}_nodp_budget${BUDGET}_seed${SEED}_hard.pt"
        if [ ! -f "${STOLEN_HARD}" ]; then
            echo "[${TASK}] Stealing (hard, no DP) | budget=${BUDGET}, seed=${SEED}..."
            python steal_prompt.py \
                --task ${TASK} \
                --victim_ckpt ${VICTIM_CKPT} \
                --out_ckpt ${STOLEN_HARD} \
                --budget ${BUDGET} \
                --epochs ${EPOCHS} \
                --batch_size ${BATCH_SIZE_TRAIN} \
                --lr ${LR} \
                --seed ${SEED} \
                --oracle hard
            echo "✓ Stolen (hard, no DP) saved"
        else
            echo "✓ Stolen (hard, no DP) already exists"
        fi
        
        # --- SOFT ORACLE (DP) ---
        STOLEN_DP_SOFT="${CKPT_DIR}/stolen_${TASK}_dp_eps${EPSILON}_budget${BUDGET}_seed${SEED}_soft.pt"
        if [ ! -f "${STOLEN_DP_SOFT}" ]; then
            echo "[${TASK}] Stealing (soft, DP) | budget=${BUDGET}, seed=${SEED}..."
            python steal_prompt.py \
                --task ${TASK} \
                --victim_ckpt ${VICTIM_DP_CKPT} \
                --out_ckpt ${STOLEN_DP_SOFT} \
                --budget ${BUDGET} \
                --epochs ${EPOCHS} \
                --batch_size ${BATCH_SIZE_TRAIN} \
                --lr ${LR} \
                --seed ${SEED} \
                --oracle probs
            echo "✓ Stolen (soft, DP) saved"
        else
            echo "✓ Stolen (soft, DP) already exists"
        fi
        
        # --- HARD ORACLE (DP) ---
        STOLEN_DP_HARD="${CKPT_DIR}/stolen_${TASK}_dp_eps${EPSILON}_budget${BUDGET}_seed${SEED}_hard.pt"
        if [ ! -f "${STOLEN_DP_HARD}" ]; then
            echo "[${TASK}] Stealing (hard, DP) | budget=${BUDGET}, seed=${SEED}..."
            python steal_prompt.py \
                --task ${TASK} \
                --victim_ckpt ${VICTIM_DP_CKPT} \
                --out_ckpt ${STOLEN_DP_HARD} \
                --budget ${BUDGET} \
                --epochs ${EPOCHS} \
                --batch_size ${BATCH_SIZE_TRAIN} \
                --lr ${LR} \
                --seed ${SEED} \
                --oracle hard
            echo "✓ Stolen (hard, DP) saved"
        else
            echo "✓ Stolen (hard, DP) already exists"
        fi
        
        echo ""
        echo "================================================"
        echo "SEED ${SEED} | BUDGET ${BUDGET} - Evaluation"
        echo "================================================"
        
        # Evaluate SOFT oracle
        echo "[${TASK}] Evaluating soft oracle results..."
        python eval.py \
            --task ${TASK} \
            --victim_ckpt ${VICTIM_CKPT} \
            --stolen_ckpt ${STOLEN_SOFT} \
            --victim_dp_ckpt ${VICTIM_DP_CKPT} \
            --stolen_dp_ckpt ${STOLEN_DP_SOFT} \
            --output_dir ${RESULTS_DIR} \
            --budget ${BUDGET} \
            --seed ${SEED} \
            --oracle probs \
            --batch_size ${BATCH_SIZE_EVAL}
        
        # Evaluate HARD oracle
        echo "[${TASK}] Evaluating hard oracle results..."
        python eval.py \
            --task ${TASK} \
            --victim_ckpt ${VICTIM_CKPT} \
            --stolen_ckpt ${STOLEN_HARD} \
            --victim_dp_ckpt ${VICTIM_DP_CKPT} \
            --stolen_dp_ckpt ${STOLEN_DP_HARD} \
            --output_dir ${RESULTS_DIR} \
            --budget ${BUDGET} \
            --seed ${SEED} \
            --oracle hard \
            --batch_size ${BATCH_SIZE_EVAL}
        
        echo "✓ Evaluations complete for budget ${BUDGET}, seed ${SEED}"
    done
    
    echo ""
    echo "================================================"
    echo "Completed all budgets for SEED ${SEED}"
    echo "================================================"
    echo ""
done

echo ""
echo "=========================================="
echo "TASK ${TASK} COMPLETE!"
echo "Results saved to: ${RESULTS_DIR}"
echo "=========================================="
