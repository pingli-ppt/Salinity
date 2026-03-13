python3 ./scripts/run_benchmark.py \
--config-path "1.json" \
--data-name-list "1.csv" \
--strategy-args '{"horizon": 360, "target_channel": [-1]}' \
--model-name "GCGNet" \
--model-hyper-params '{
"batch_size": 4,
"d_ff": 128,
"d_model": 64,
"dropout": 0,
"pred_len": 24,
"loss": "MAE",
"lr": 0.0001,
"lradj": "type3",
"n_heads": 2,
"norm": true,
"num_epochs": 1,
"patch_len": 24,
"patience": 2,
"rank": 2,
"seq_len": 48
}' \
--gpus 0 \
--num-workers 1 \
--timeout 60000 \
--save-path "test/debug/GCGNet"