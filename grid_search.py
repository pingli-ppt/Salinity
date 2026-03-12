import os
import json
import subprocess
import numpy as np
from itertools import product
import sys

# 1. 网格搜索参数（需调优的参数）
PARAM_GRID = {
    "dropout": [0.1, 0.2, 0.3 , 0.4],          
    "fc_dropout": [0.1, 0.2, 0.3 , 0.4],
    "lr": [0.0001, 0.0005, 0.001],
    "batch_size": [64, 128]
}

# 2. 固定参数
FIXED_PARAMS = {
    "CI": 1, "d_ff": 1024, "d_model": 512, "e_layers": 2, "factor": 3,
    "horizon": 24, "k": 2, "loss": "MAE", "lradj": "type1", "n_heads": 1,
    "norm": True, "num_epochs": 100, "num_experts": 4, "patch_len": 48,
    "patience": 5, "seq_len": 144
}

# 3. 运行配置
RUN_CONFIG = {
    "config_path": "rolling_forecast_config.json",  
    "data_name": "ocean_salinity_duet.csv",
    "gpus": 0,
    "num_workers": 8,
    "save_path_prefix": "ocean_salinity_duet/DUET_grid_search"
}

# 调用run_benchmark.py
def grid_search_duet():
    # 1. 生成所有参数组合
    param_names = list(PARAM_GRID.keys())
    param_values = list(PARAM_GRID.values())
    all_combinations = list(product(*param_values))

    # 2. 初始化最优结果
    best_mae_norm = float("inf")
    best_params = None
    results = []

    # 3. 遍历参数组合
    for idx, combo in enumerate(all_combinations):
        # 合并固定参数和当前调优参数
        current_params = FIXED_PARAMS.copy()
        current_params.update(dict(zip(param_names, combo)))
        save_path = f"{RUN_CONFIG['save_path_prefix']}_params_{idx+1}"

        # 4. 构造命令行参数
        model_hyper_params = json.dumps(current_params).replace('"', '\\"')
        strategy_args = json.dumps({"horizon": 24}).replace('"', '\\"')
        cmd = (
            f"python3 ./scripts/run_benchmark.py "
            f"--config-path \"{RUN_CONFIG['config_path']}\" "
            f"--data-name-list \"{RUN_CONFIG['data_name']}\" "
            f"--strategy-args \"{strategy_args}\" "
            f"--model-name \"duet.DUET\" "
            f"--model-hyper-params \"{model_hyper_params}\" "
            f"--deterministic \"full\" "
            f"--gpus {RUN_CONFIG['gpus']} "
            f"--num-workers {RUN_CONFIG['num_workers']} "
            f"--timeout 60000 "
            f"--save-path \"{save_path}\""
        )

        # 5. 执行命令
        try:
            result = subprocess.run(
                cmd, shell=True, check=True,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                encoding="utf-8"
            )

        except subprocess.CalledProcessError as e:
            print(f"参数组合运行失败：{e.stderr}\n")
            continue

    print("===== 网格搜索完成 =====")
    
if __name__ == "__main__":
    grid_search_duet()