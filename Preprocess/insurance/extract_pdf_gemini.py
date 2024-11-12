import os
import time
import csv
import google.generativeai as genai
from dotenv import load_dotenv
import re
import sys

dotenv_path = "../../.env"
load_dotenv(dotenv_path=dotenv_path)
# Load environment variables from .env file
#load_dotenv()

def upload_to_gemini(path, mime_type="application/pdf"):
    """Uploads the given file to Gemini."""
    file = genai.upload_file(path, mime_type=mime_type)
    print(f"Uploaded file '{file.display_name}' as: {file.uri}")
    return file

def wait_for_files_active(files):
    """Waits for the given files to be active."""
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
    """Configures the model with the provided API key."""
    global model
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(
        model_name="gemini-1.5-pro-002",
        generation_config=generation_config,
    )

def process_and_save_files(start_file_num=1, input_folder="1_input_pdfs", output_folder="gemini_pro"):
    """处理输入文件夹中的每个 PDF 文件，并将结果保存到输出文件夹。"""
    os.makedirs(output_folder, exist_ok=True)
    
    # 获取输入文件夹中所有的 PDF 文件
    all_pdfs = [f for f in os.listdir(input_folder) if f.lower().endswith('.pdf')]

    # 定义一个函数，从文件名中提取数字部分
    def get_numeric_part(filename):
        base = os.path.splitext(filename)[0]
        try:
            return int(base)
        except ValueError:
            # 如果文件名不是数字，将其放在列表的末尾
            return float('inf')

    # 根据文件名的数字部分对文件进行排序
    all_pdfs.sort(key=get_numeric_part)
    target_pdfs = all_pdfs[start_file_num - 1:]

    for pdf_file in target_pdfs:
        output_path = os.path.join(output_folder, f"{os.path.splitext(pdf_file)[0]}.txt")
        
        # 如果输出文件已存在，则跳过
        if os.path.exists(output_path):
            continue
          
        file_path = os.path.join(input_folder, pdf_file)
        try:
            file = upload_to_gemini(file_path)
            # 等待文件处理完成
            wait_for_files_active([file])
        except Exception as e:
            print(f"Error processing '{pdf_file}': {e}")
            continue
        
        # 定义提示信息
        prompt = (
            "請從 PDF 文件中提取所有文字內容，並保持文本的原始格式，以確保後續檢索的精準度。"
            "請將中文數字轉換為阿拉伯數字。以下是一些範例轉換：\n"
            "- '達一○○元' 轉為 '達100元'\n"
            "- '於一仟元' 轉為 '於1000元'\n"
            "- '足一百美元' 轉為 '足100美元'\n"
            "- '一百零九歲' 轉為 '109歲'\n"
            "- '百分之二百一十' 轉為 '210%'\n"
            "- '百分之一點三五' 轉為 '1.35%'\n"
            "- '美元三千元' 轉為 '美元3000元'\n"
            "- '民國一百零九年' 轉為 '民國109年'\n"
            "- '二月三日' 轉為 '2月3日'\n"
            "- '第二十七條' 轉為 '第27條'\n"
            "- 移除行首的條目編號格式，例如 '一、'、'二、' 等格式\n"
            "不需提取頁碼資訊。"
        )


        
        # 开始聊天会话以进行提取
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
        
        # 获取响应
        response = chat_session.send_message("請提取檔案的內容")
        
        # 保存响应文本
        with open(output_path, "w", encoding="utf-8") as txt_file:
            txt_file.write(response.text)
        
        print(f"Processed and saved result for '{pdf_file}' as '{output_path}'")
       

if __name__ == "__main__":

    
    # Retrieve the API key index from command-line arguments

    api_key_env_var = f"GEMINI_API_KEY"
    api_key = os.getenv(api_key_env_var)
    
    if api_key is None:
        print(f"Error: Environment variable '{api_key_env_var}' not found.")
        sys.exit(1)

    start_file_num = 1
    configure_model(api_key)
    process_and_save_files(start_file_num=start_file_num)