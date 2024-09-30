# Benchmark LLM Training

Repo for benchmarking LLM training workloads.


## Usage

```bash
NAME
	train.py

SYNOPSIS
	train.py CFG_PATH <flags>

POSITIONAL ARGUMENTS
	CFG_PATH
		Type: str

FLAGS
	-b, --bsz=BSZ
		Type: int
		Default: 8
	--n_workers=N_WORKERS
		Type: int
		Default: 8
	--n_steps=N_STEPS
		Type: int
		Default: 64
	-g, --grad_acc_steps=GRAD_ACC_STEPS
		Type: int
		Default: 4
	-c, --ckpt_freq=CKPT_FREQ
		Type: int
		Default: 16
	-p, --pt_compile=PT_COMPILE
		Type: bool
		Default: False
	-o, --output_dir=OUTPUT_DIR
		Type: str
		Default: 'outputs/'

NOTES
	You can also use flags syntax for POSITIONAL ARGUMENTS
```
