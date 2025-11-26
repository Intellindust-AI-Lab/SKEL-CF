#!/bin/bash
NPZ_FILE=''
OUT_DIR=''


PYTHONPATH=. python skelfitter/run_fit.py \
            --out_dir "$OUT_DIR" \
            --smpl_data_path "$NPZ_FILE"
