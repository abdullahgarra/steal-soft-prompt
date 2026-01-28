# Soft Prompt Stealing with DP Protection - Experimental Pipeline

## Overview

This pipeline systematically evaluates soft prompt stealing attacks across multiple GLUE tasks with and without differential privacy protection.

## Directory Structure

```
.
├── train_victim.py          # Train victim model (no DP)
├── train_victim_dp.py       # Train victim model (with DP-SGD)
├── steal_prompt.py          # Black-box prompt stealing attack
├── eval.py                  # Evaluate models and save results to CSV
├── analyze_results.py       # Aggregate results across seeds
├── run_experiments.sh       # SLURM array job script
├── utils.py                 # Shared utilities (SoftPromptT5)
├── checkpoints/             # Model checkpoints organized by task
│   ├── sst2/
│   ├── qqp/
│   ├── mnli/
│   └── qnli/
├── results/                 # CSV results organized by task
│   ├── sst2/
│   ├── qqp/
│   ├── mnli/
│   └── qnli/
└── logs/                    # SLURM output logs
```

## Experimental Design

### Parameters
- **Tasks**: SST-2, QQP, MNLI, QNLI
- **Seeds**: 42, 43, 44 (for averaging)
- **Budgets**: 200, 500, 1000, 5000 (query budget for stealing)
- **Oracles**: soft (probability outputs), hard (argmax labels)
- **DP Settings**: ε=8.0, δ=1/|D_s|
- **Training**: 5 epochs for victims, 20 epochs for stealing

### Workflow

Each SLURM array task (0-3) processes one dataset through:

1. **Train Victims** (once per seed)
   - Standard victim model (no DP)
   - DP-protected victim model (ε=8.0)

2. **Stealing Attacks** (for each seed × budget combination)
   - Soft oracle from standard victim
   - Hard oracle from standard victim
   - Soft oracle from DP victim
   - Hard oracle from DP victim

3. **Evaluation** (after each stealing)
   - Accuracy, F1 score, confidence
   - Agreement rate between victim and stolen
   - KL divergence (victim || stolen)
   - Results appended to CSV files

## Usage

### 1. Submit SLURM Array Job

```bash
# Make script executable
chmod +x run_experiments.slurm

# Submit array job (runs 4 tasks in parallel, one per dataset)
sbatch run_experiments.slurm
```

The script will:
- Create `logs/` directory for output
- Run all seed × budget combinations
- Save checkpoints incrementally (won't rerun if exists)
- Append results to CSV files after each evaluation

### 2. Monitor Progress

```bash
# Check job status
squeue -u $USER

# Watch logs in real-time
tail -f logs/run_<jobid>_<array_id>.out

# Array task 0 = sst2
# Array task 1 = qqp
# Array task 2 = mnli
# Array task 3 = qnli
```
### 3. Instructions for Notebook Usage
[TODO]

### 4. Access Raw Results

CSV files are organized as:
```
results/{task}
├── results_nodp_probs.csv    # Standard victim, soft oracle
├── results_nodp_hard.csv     # Standard victim, hard oracle
├── results_dp_probs.csv      # DP victim, soft oracle
├── results_dp_hard.csv       # DP victim, hard oracle
```

Each row contains:
- Experimental params: task, seed, budget, oracle, dp
- Victim metrics: acc, f1, avg_conf, pos_rate, prompt_norm
- Stolen metrics: acc, f1, avg_conf, pos_rate, prompt_norm
- Pair metrics: agreement, avg_kl_victim_to_stolen, prompt_l2_dist, prompt_cos_sim

## Manual Runs

### Train a victim model
```bash
python train_victim.py --task sst2 --epochs 5 --batch_size 256 --seed 42
```

### Train a DP victim
```bash
python train_victim_dp.py --task sst2 --epochs 5 --batch_size 128 --epsilon 8.0 --seed 42
```

### Run stealing attack
```bash
python steal_prompt.py \
  --task sst2 \
  --victim_ckpt checkpoints/sst2/victim_sst2_P20_seed42.pt \
  --out_ckpt checkpoints/sst2/stolen_sst2_nodp_budget1000_seed42_soft.pt \
  --budget 1000 \
  --epochs 20 \
  --oracle probs \
  --seed 42
```

### Evaluate models
```bash
python eval.py \
  --task sst2 \
  --victim_ckpt checkpoints/sst2/victim_sst2_P20_seed42.pt \
  --stolen_ckpt checkpoints/sst2/stolen_sst2_nodp_budget1000_seed42_soft.pt \
  --victim_dp_ckpt checkpoints/sst2/victim_dp_sst2_eps8.0_P20_seed42.pt \
  --stolen_dp_ckpt checkpoints/sst2/stolen_sst2_dp_eps8.0_budget1000_seed42_soft.pt \
  --budget 1000 \
  --oracle probs \
  --seed 42
```

## Key Metrics

### Individual Model Metrics
- **acc**: Classification accuracy
- **f1**: F1 score (binary tasks only)
- **avg_conf**: Average prediction confidence
- **pos_rate**: Positive class rate (binary tasks)
- **prompt_norm**: L2 norm of soft prompt

### Pair Metrics (Victim vs Stolen)
- **agreement**: Fraction of examples where predictions match
- **avg_kl_victim_to_stolen**: KL divergence measuring distribution similarity
- **prompt_cos_sim**: Cosine similarity between prompt embeddings
- **prompt_l2_dist**: L2 distance between prompts
