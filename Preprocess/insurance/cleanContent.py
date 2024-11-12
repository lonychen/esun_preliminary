import os
import re
import argparse

def search_and_replace_patterns_in_files(folder_path, patterns, replacements, replace=False, output_folder="clean_gemini"):
    # 確保目錄存在
    if not os.path.isdir(folder_path):
        print("資料夾不存在！")
        return
    
    # 若為替換模式，確保替換模式和替換內容數量相符
    if replace and len(replacements) != len(patterns):
        print("替換模式和替換內容數量不相符！")
        return
    
    # 若為替換模式，建立輸出資料夾
    if replace and not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # 將目錄中的檔案依照檔名數字排序
    files = sorted(
        [f for f in os.listdir(folder_path) if f.endswith('.txt')],
        key=lambda x: int(x.split('.')[0])
    )
    
    results = {}
    
    # 遍歷每個檔案
    for file in files:
        file_path = os.path.join(folder_path, file)
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 處理每個模式
        file_results = []
        for i, pattern in enumerate(patterns):
            # 如果選擇替換模式
            if replace:
                # 替換符合模式的內容
                content = re.sub(pattern, replacements[i], content)
            else:
                # 找出符合模式的片段
                matches = re.findall(pattern, content)
                file_results.extend([f"...{match}..." for match in matches])
        
        # 儲存結果
        if replace:
            output_path = os.path.join(output_folder, file)  # 將替換後的內容儲存到 clean_gemini 資料夾
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(content)
        elif file_results:
            results[file] = file_results  # 存符合條件的片段
    
    # 印出結果（僅搜尋模式時）
    if not replace:
        for file, result in results.items():
            print(f"[{file}]")
            for context in result:
                print(context)
            print()  # 空行分隔每個檔案的輸出

if __name__ == "__main__":
    # 設定搜尋模式和替換文字
    patterns = [
        r'.{0,15}南山.{0,50}', 
        r'.{0,15}凱基.{0,50}', 
        r'.{0,15}全球人壽.{0,50}', 
        r'.{0,15}安達人壽.{0,50}', 
        r'.{0,15}民國.{0,50}',
        r'2023年', 
        r'2022年',
        r'2021年', 
        r'2020年',
        r'2019年', 
        r'2018年',
        r'2017年', 
        r'1999年',
       
    ]  # 搜尋模式陣列

    replacements = [
        '', 
        '', 
        '', 
        '',
        '',
        '112年', 
        '111年', 
        '110年', 
        '109年',
        '108年', 
        '107年', 
        '106年', 
        '88年'
    ]  # 替換的文字（與模式對應）
    
    output_folder = 'clean_gemini'  # 替換後的輸出資料夾
    
    # 設定命令列引數
    parser = argparse.ArgumentParser(description="搜尋並替換 txt 檔案中的模式")
    parser.add_argument("folder_path", type=str, help="資料夾路徑")
    parser.add_argument("--replace", action="store_true", help="是否替換匹配的模式並儲存到 clean_gemini 資料夾")

    args = parser.parse_args()

    # 呼叫主函數並傳入命令列引數
    search_and_replace_patterns_in_files(
        args.folder_path,
        patterns,
        replacements,
        replace=args.replace,
        output_folder=output_folder
    )
