import os
import time
import csv
import google.generativeai as genai
from dotenv import load_dotenv
import re
import sys

dotenv_path = "../../.env"
load_dotenv(dotenv_path=dotenv_path)


def upload_to_gemini(path, mime_type="application/pdf"):
    """上傳指定文件到 Gemini 平台。

    Args:
        path (str): 文件路徑。
        mime_type (str): 文件的 MIME 類型，預設為 PDF 格式。

    Returns:
        file: 上傳後的文件物件。
    """
    file = genai.upload_file(path, mime_type=mime_type)
    print(f"Uploaded file '{file.display_name}' as: {file.uri}")
    return file

def wait_for_files_active(files):
    """等待指定的文件狀態變為 active。

    Args:
        files (list): 要等待處理完成的文件列表。
    """
    print("Waiting for file processing...")
    for file in files:
        while file.state.name == "PROCESSING":
            print(".", end="", flush=True)
            time.sleep(10)
            file = genai.get_file(file.name)
        if file.state.name != "ACTIVE":
            raise Exception(f"File {file.name} failed to process")
    print("...all files ready")

# Initialize model configuration
generation_config = {
    "temperature": 0,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

model = None

def configure_model(api_key):
    """配置生成模型，設定 API 金鑰和生成參數。

    Args:
        api_key (str): 用於配置生成模型的 API 金鑰。
    """
    global model
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(
        model_name="gemini-1.5-pro-002",
        generation_config=generation_config,
    )

def load_target_pdfs(csv_file="finance_mapping_sort.csv"):
    """根據 CSV 文件內容載入目標 PDF 文件的名稱和報告類型。

    Args:
        csv_file (str): 包含 PDF 檔名和類型的 CSV 文件路徑。

    Returns:
        list: 包含 (pdf_name, report_type) 元組的列表。
    """
    target_pdfs = []
    with open(csv_file, "r", encoding="utf-8") as file:
        reader = csv.reader(file)
        for row in reader:
            pdf_name = row[0]
            report_type = row[-1]
            target_pdfs.append((pdf_name, report_type))
    return target_pdfs

def process_and_save_files(start_file_num=1, input_folder="pdf", output_folder="gemini_pro3"):
    """處理指定資料夾中的每個 PDF 文件，並將結果儲存在輸出資料夾中。

    Args:
        start_file_num (int): 開始處理的文件編號。
        input_folder (str): 輸入文件的資料夾。
        output_folder (str): 儲存處理後文件的資料夾。
    """
    os.makedirs(output_folder, exist_ok=True)
    
    # Load only the target PDF files based on CSV
    target_pdfs = load_target_pdfs()
    
    # Sort the target PDFs by numeric order
    target_pdfs = sorted(
        target_pdfs,
        key=lambda x: int(re.findall(r'\d+', x[0])[0]) if re.findall(r'\d+', x[0]) else float('inf')
    )
    
    # Filter to only include files starting from start_file_num
    start_index = next((i for i, f in enumerate(target_pdfs) if int(re.findall(r'\d+', f[0])[0]) >= start_file_num), None)
    target_pdfs = target_pdfs[start_index:] if start_index is not None else []

    for pdf_file, report_type in target_pdfs:
        output_path = os.path.join(output_folder, f"{os.path.splitext(pdf_file)[0]}.txt")
        
        # Skip if output file already exists
        if os.path.exists(output_path):
            #print(f"Skipping '{pdf_file}' as output already exists.")
            continue
          
        file_path = os.path.join(input_folder, pdf_file)
        
        # Check if the PDF file exists before uploading
        if not os.path.exists(file_path):
            print(f"Skipping '{pdf_file}' as the file does not exist.")
            continue
            
        file = upload_to_gemini(file_path)
        
        # Wait until the file is active
        wait_for_files_active([file])
        
        # Choose the prompt based on report type
        if report_type == "合併現金流量表":
            prompt = (
                "這是一份現金流量表，共有4個欄位，代碼、項目、本期金額、上期金額，"
                "提取檔案中的內容後，輸出下列資訊：XX{項目}為{本期金額}。YY{項目}為{上期金額}。"
                "請完整列出所有項目的金額，不用其他資訊。數字如果有括弧，表示是負數，改用-表示。"
                "請跳過pdf的前兩頁不用提取。"
            )
        elif report_type == "合併綜合損益表":
            prompt = (
                "這是一份合併綜合損益表，每一列共有10個欄位，分別是：代碼、項目、本期金額、本期金額所佔百分比、去年同期金額、去年同期金額百分比、今年金額、今年金額百分比、去年金額、去年金額百分比。"
                "提取檔案中的{項目}、{本期金額}、{去年同期金額}的欄位資訊後，以下列格式輸出：XX{項目}為{本期金額}。YY{項目}為{去年同期金額}。"
                "不需要{項目}裡的附註，{項目}、{本期金額}、{去年同期金額}的輸出格式請保留{}。"
                "請完整列出所有項目的資料，不用其他資訊。"
                "數字如果有括弧，表示是負數，改用-表示。"
                "請跳過pdf的前兩頁不用提取。"
            )
        elif report_type == "合併資產負債表":
            prompt = (
                "這是一份合併資產負債表，每一列共有9個欄位，分別是：代碼、項目、附註、本期金額、本期金額所佔百分比、去年金額、去年金額百分比、去年同期金額、去年同期金額百分比。"
                "提取檔案中的{項目}、{本期金額}、{去年同期金額}的欄位資訊後，以下列格式輸出：XX{項目}為{本期金額}。YY{項目}為{去年同期金額}。"
                "不需要{項目}裡的附註，{項目}、{本期金額}、{去年同期金額}的輸出格式請保留{}。"
                "請完整列出所有項目的資料，不用其他資訊。"
                "數字如果有括弧，表示是負數，改用-表示。"
                "請跳過pdf的前兩頁不用提取。"
            )
        elif report_type == "合併權益變動表":
            prompt = (
                "這是一份合併權益變動表，請將表格轉換為文字敘述，遵循以下格式並保持一致的條理性。目的是為了檢索方便，請您簡明且分層次地描述表格中的每個主要項目及其對應數據。"
                "格式要求如下："
                "對於每個時期（例如「期初餘額」、「年度調整及分配」和「期末餘額」），請依次列出相關條目，並在條目名稱後使用冒號，以數據跟隨說明。"
                "對於每個子項目，請使用「項目名稱：金額」的格式，清楚地顯示表格中的數據。"
                "將相同類型的項目（例如「淨利」、「其他綜合損益」等）歸納在同一段落內，以利快速檢索。"
                "數字分隔符號以千元為單位，數字如果有括弧，表示是負數，改用-表示。"
                "年份如果用中文字的方式表示時，例如「民國一一一年」，請改成「111年」。"
                "格式範例(請依此範例進行)："
                "**期初餘額(111年1月1日)**"
                " - 普通股股本：1,000,000。"
                " - 資本公積：1,000,000。"
                " - 法定盈餘公積：1,000,000。"
                " ..."
                "**年度調整及分配**"
                " - 發放予子公司股利調整資本公積：1,000,000。"
                " - 法定盈餘公積：1,000,000。"
                " - 特別盈餘公積：1,000,000。"
                " ..."
                 "**期末餘額(111年9月30日)**"
                " - 普通股股本：1,000,000。"
                " - 資本公積：1,000,000。"
                " - 法定盈餘公積：1,000,000。"
                " ..."
                "請依照此格式準確描述表格，保持條目、金額的標準格式，並確保所有數據和段落分層的清晰度，以便後續檢索模型進行精準匹配。"
                "請跳過pdf的前兩頁不用提取。"
            )
        elif report_type == "會計師核閱報告":
            prompt = (
                "這是一份會計師核閱報告，請直接提取「前言」、「範圍」、「保留結論之基礎」、「保留結論」和「其他事項」等段落的原始內容，保持文字不變，不進行任何修改，並按照以下格式進行分層描述，以便後續檢索："
                "- 前言：..."
                "- 範圍：..."
                "- 保留結論之基礎：..."
                "- 保留結論：..."
                "- 其他事項（如果有）：..."
                "- 其他資訊：包括報告的日期、會計師事務所名稱、簽署的會計師姓名，以及金融監督管理委員會核准文號（如有）。"
                "範例格式（請依此範例進行）："
                "*前言*： [原始文字無修改]"
                "*範圍*： [原始文字無修改]"
                "*保留結論之基礎*： [原始文字無修改]"
                "*保留結論*： [原始文字無修改]"
                "*其他事項*： [原始文字無修改]（若適用）"
                "*其他資訊*：勤業眾信聯合會計師事務所，會計師鍾鳴遠及林政治，金融監督管理委員會核准文號：金管證審字第 1050024633 號，報告日期為112年11月9日。"
                "請按照此格式提取內容，確保每個段落的文字保持與原始報告一致，以便後續精準檢索。"
                "請跳過pdf的前兩頁不用提取。"
            )
        else:
            prompt = (
                "請從 PDF 文件中提取所有文字，並保持與原始文本的格式一致，以便後續進行精準檢索。"
                "若 PDF 中包含表格，請完全避免使用表格格式進行描述。每個項目及其對應數據分行顯示，按以下格式描述："
                "1. 數字分隔符號以千元為單位，數字如果有括弧，表示是負數，改用-表示。"
                "2. 將表格中不同時期的數據分段描述，每段標註明確的時間。例如："
                "   - 期初餘額（111年1月1日）："
                "     - 普通股股本：1,000,000"
                "     - 資本公積：1,000,000"
                "     - 法定盈餘公積：1,000,000"
                "3. 關係人名稱及關係的描述範例（避免表格，將同類型的列在一起，注意「關係」可能會用符號代表跟上一筆相同）："
                "   - 關係人名稱及關係："
                "     - 關聯企業：Intelligo Technology Inc、賽微科技(股)公司、..."
                "     - 主要管理階層控制之個體：亞信電子(股)公司、賽賽科技(股)公司、..."
                "     - 主要管理階層：亞電子公司、賽技公司、..."
                "     - ..."
                "4. 完全避免使用表格排版，請直接列出每個項目。示例："
                "   - 111年9月30日 強制透過損益按公允價值衡量-衍生工具(未指定避險)-遠期外匯合約：1,457"
                "   - 110年12月31日 強制透過損益按公允價值衡量-非衍生金融資產-上市(櫃)股票：170,417"
                "5. 將中文年份改為阿拉伯數字，例如「民國一一一年」轉換為「111年」。"
                "請跳過pdf的前兩頁不用提取。"
            )


        # Start chat session for extraction
        chat_session = model.start_chat(
            history=[
                {
                    "role": "user",
                    "parts": [
                        file,
                        prompt,
                    ],
                }
            ]
        )
        
        # Retrieve the response
        response = chat_session.send_message("請提取檔案的內容")
        
        # Save the response text
        with open(output_path, "w", encoding="utf-8") as txt_file:
            txt_file.write(response.text)
        
        print(f"Processed and saved result for '{pdf_file}' as '{output_path}'")

if __name__ == "__main__":
   
    api_key_env_var = f"GEMINI_API_KEY"
    api_key = os.getenv(api_key_env_var)
    
    if api_key is None:
        print(f"Error: Environment variable '{api_key_env_var}' not found.")
        sys.exit(1)

    start_file_num = 1
    configure_model(api_key)
    process_and_save_files(start_file_num=start_file_num)