import pandas as pd
import os

def create_generative_db(input_csv_path='./data/supermarket_dataset_real_scene/statistics.csv', output_dir='./src/db'):
    """
    讀取包含生成影像資訊的 CSV，並創建一個標準化的資料庫 CSV 檔案。

    Args:
        input_csv_path (str): 來源 CSV 檔案的路徑。
        output_dir (str): 輸出目錄的路徑。
    """
    output_csv_name = 'gen_db.csv'
    output_csv_path = os.path.join(output_dir, output_csv_name)

    # --- 1. 讀取來源 CSV ---
    try:
        print(f"正在讀取來源檔案: {input_csv_path}...")
        df = pd.read_csv(input_csv_path)
        print("來源檔案讀取成功。")
    except FileNotFoundError:
        print(f"錯誤：找不到來源檔案 '{input_csv_path}'。請確認檔案路徑是否正確。")
        return
    except Exception as e:
        print(f"讀取 CSV 時發生錯誤: {e}")
        return

    # --- 2. 驗證必要欄位 ---
    # !!! 如果您的欄位名稱不同，請在此處修改 !!!
    original_path_col = 'filename'
    category_col = 'category'
    
    if original_path_col not in df.columns or category_col not in df.columns:
        print(f"錯誤：CSV 檔案中缺少必要的欄位。")
        print(f"請確保檔案包含 '{original_path_col}' 和 '{category_col}' 這兩個欄位。")
        print(f"目前找到的欄位有: {df.columns.tolist()}")
        return

    # --- 3. 處理數據 ---
    print("正在生成唯一 ID 和新路徑...")
    # 生成唯一 ID (從 0 開始)
    df['id'] = range(len(df))

    # 定義一個函式來建構新路徑
    def construct_unique_path(row):
        # 從原始路徑中提取檔案名稱
        filename = os.path.basename(row[original_path_col])
        # 獲取類別
        category = row[category_col]
        # 組合新路徑
        # 使用 os.path.join 確保路徑分隔符的跨平台相容性
        new_path = os.path.join('./data/supermarket_dataset_real_scene', str(category), filename).replace('\\', '/')
        return new_path

    # 應用函式以創建新路徑欄位
    df['unique_path'] = df.apply(construct_unique_path, axis=1)
    print("新路徑生成完畢。")

    # --- 4. 建立輸出的 DataFrame ---
    # 只選擇我們需要的欄位
    output_df = df[['id', 'unique_path']]

    # --- 5. 儲存結果到新的 CSV ---
    try:
        # 確保輸出目錄存在
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"正在將結果儲存至: {output_csv_path}...")
        output_df.to_csv(output_csv_path, index=False, encoding='utf-8')
        print("✅ 檔案已成功儲存！")
        
        print("\n--- 輸出檔案預覽 (前5筆) ---")
        print(output_df.head())
        print("------------------------------")

    except Exception as e:
        print(f"儲存 CSV 時發生錯誤: {e}")

create_generative_db()