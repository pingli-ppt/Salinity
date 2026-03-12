import os
import json
import pandas as pd

# ===================== 路径 =====================
RESULT_FOLDER = "result/ETTh1/DUET"
OUTPUT_EXCEL = "实验结果总表_永久保存.xlsx"
OUTPUT_CSV   = "实验结果总表_永久保存.csv"
# ====================================================

def process_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = [x.strip() for x in f if x.strip()]

        # 解析模型参数
        line1 = lines[0]
        start = line1.find('DUET;{') + 5
        end = line1.rfind('}') + 1
        json_str = line1[start:end].replace('""', '"')
        model_dict = json.loads(json_str)
        model_dict["model"] = "DUET"
        model_dict["source_file"] = os.path.basename(file_path)

        # 解析指标
        metrics = {}
        for line in lines[1:]:
            parts = line.split(',')
            metric_name = parts[-2].strip().strip('"')
            metric_value = float(parts[-1].strip())
            metrics[metric_name] = metric_value

        return {**model_dict, **metrics}
    except:
        return None

# ===================== 主程序：自动追加，不覆盖 =====================
if __name__ == "__main__":
    # 1. 读取已有的历史数据（如果存在）
    existing_data = []
    if os.path.exists(OUTPUT_EXCEL):
        df_exist = pd.read_excel(OUTPUT_EXCEL)
        existing_data = df_exist.to_dict('records')
        print(f"✅ 已加载历史数据：{len(existing_data)} 条")

    # 2. 读取当前 result 文件夹里的新数据
    new_data = []
    csv_files = [f for f in os.listdir(RESULT_FOLDER) if f.startswith("test_report") and f.endswith(".csv")]

    for fname in csv_files:
        row = process_file(os.path.join(RESULT_FOLDER, fname))
        if row:
            new_data.append(row)

    print(f"📊 当前文件夹新数据：{len(new_data)} 条")

    # 3. 合并（自动去重，不重复添加同文件数据）
    existing_files = {r.get("source_file") for r in existing_data}
    final_data = existing_data + [r for r in new_data if r.get("source_file") not in existing_files]

    # 4. 保存（真正的追加模式，不覆盖！）
    df = pd.DataFrame(final_data)
    df.to_excel(OUTPUT_EXCEL, index=False, engine="openpyxl")
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    print(f"\n🎉 保存成功！总数据：{len(final_data)} 条")
    print(f"📁 文件已永久保存，不会被覆盖！")