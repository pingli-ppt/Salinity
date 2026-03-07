import pandas as pd
import numpy as np

# 定义数据集路径（和项目要求一致）
data_path = "dataset/forecasting/ETTh1.csv"
# 读取数据集（保留所有列，包括之前添加的cols）
df = pd.read_csv(data_path)

# 核心：将date列从字符串转为datetime格式
df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d %H:%M:%S")

# 校验时间间隔是否为1小时（ETTh1原始数据是严格1小时，这里打印结果确认）
time_diff = df["date"].diff().dropna()
is_1h = all(time_diff == pd.Timedelta(hours=1))
print(f"时间列转换完成，是否为严格1小时间隔：{is_1h}")
print(f"数据集总行数（含表头）：{len(df)+1}，数据行数：{len(df)}")

# 覆盖保存处理后的数据集（确保路径不变，代码能正常读取）
df.to_csv(data_path, index=False, encoding="utf-8")
print(f"预处理完成，已覆盖保存到 {data_path}")
