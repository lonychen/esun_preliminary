import os
import re
import argparse

def search_and_replace_patterns_in_files(folder_path, patterns, replacements, replace=False, output_folder="clean_gemini"):
    """在指定的資料夾中搜尋並替換文件中的模式。

    Args:
        folder_path (str): 資料夾路徑。
        patterns (list): 正則表達式模式的列表。
        replacements (list): 用於替換的內容列表。
        replace (bool): 是否替換匹配的模式。若為 False 則僅顯示匹配內容。
        output_folder (str): 儲存替換後文件的資料夾路徑。
    """
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
        r'一一二年', 
        r'一一一年', 
        r'一一○年', 
        r'一月一日', 
        r'七月一日', 
        r'三月三十一日', 
        r'九月三十日', 
    
        r'\s*2025\s*年', 
        r'\s*2024\s*年', 
        r'\s*2023\s*年', 
        r'\s*2022\s*年',
        r'\s*2021\s*年', 
        r'\s*2020\s*年',
        r'\s*2019\s*年', 
        r'\s*2018\s*年',
        r'\s*2017\s*年', 
        r'\s*1999\s*年',
        
        r'\s*1\s*月\s*1\s*日至\s*3\s*月\s*31\s*日',
        r'\s*4\s*月\s*1\s*日至\s*6\s*月\s*30\s*日',
        r'\s*7\s*月\s*1\s*日至\s*9\s*月\s*30\s*日',
        r'\s*10\s*月\s*1\s*日至\s*12\s*月\s*31\s*日',
        r'\s*1\s*月\s*1\s*日至\s*6\s*月\s*30\s*日',
        r'\s*1\s*月\s*1\s*日至\s*9\s*月\s*30\s*日',
        r'前三季',
        r'第一季',
        r'第二季',
        r'第三季',
        r'\s*1\s*至\s*3\s*月',
        r'\s*1\s*至\s*9\s*月',
        r'\s*7\s*至\s*9\s*月',
        
        r'透過其他綜合損益按',
        r'透過損益按',
        
        
        #r'一.{1,4}\s*年.{0,15}',
    ]  # 搜尋模式陣列

    replacements = [
        '112年', 
        '111年', 
        '110年', 
        '1月1日', 
        '7月1日', 
        '3月31日', 
        '9月30日', 
        
        '114年', 
        '113年', 
        '112年', 
        '111年', 
        '110年', 
        '109年',
        '108年', 
        '107年', 
        '106年', 
        '88年',
        
        '第1季',
        '第2季',
        '第3季',
        '第4季',
        '前2季',
        '前3季',
        '前3季',
        '第1季',
        '第2季',
        '第3季',
        '第1季',
        '前3季',
        '第3季',
        
        r'按',
        r'按',
        
    ]  # 替換的文字（與模式對應）
    
    output_folder = 'clean_gemini'  # 替換後的輸出資料夾
    
    # 設定命令列引數
    parser = argparse.ArgumentParser(description="搜尋並替換 txt 檔案中的模式")
    parser.add_argument("folder_path", nargs="?", default="gemini_pro", type=str, help="資料夾路徑，預設為 gemini_pro")
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
