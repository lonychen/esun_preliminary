# 初賽使用的方法

在初賽中，使用了以下步驟來處理檢索任務：

1. **資料預處理**

   資料預處理分為 `insurance`、`finance` 和 `faq` 三部分：

   - **insurance & finance**：
     - **insurance** 的原始 PDF 文件存放於 `1_input_pdfs` 資料夾中，而 **finance** 的原始 PDF 文件存放於 `pdf` 資料夾。
     - 使用 `extract_pdf_gemini.py` 呼叫 LLM（Gemini 1.5 Pro 版本）來提取文字，結果存放於 `gemini_pro` 資料夾。
     - 接著使用 `cleanContent.py` 進行二次處理，將清洗過的結果存放於 `clean_gemini` 資料夾。
     
     `.env` 文件中需要包含 `GEMINI_API_KEY`，以供 `extract_pdf_gemini.py` 使用。
   
   - **faq**：
     - 將原始 JSON 資料拆分成 TXT 格式，以方便統一進行處理。

2. **BM25 檢索**

   首先使用 BM25 方法從大型語料庫中檢索出最相關的前 3 筆文檔。

   程式執行方式：
   ```bash
   python bm25.py
   ```

3. **外部 LLM (grok-beta) 重新排序**

   接著使用 grok-beta 模型對檢索出的文檔進行重新排序，如果 LLM（大語言模型）認為所有檢索出的文檔都不符合查詢要求，會回傳-1。

   用法範例：
   ```bash
   python bm25_gpt.py --model grok-beta
   ```

   `.env` 文件中需要包含 `XAI_API_KEY`，以供 `bm25_gpt.py` 使用。

4. **合併答案生成**

   最後使用 `generate_comparison.py` 來生成合併答案，如果 LLM（大語言模型）認為所有檢索出的文檔都不符合查詢要求(-1)，則選擇 BM25 的最佳答案作為最終結果，最終合併答案會輸出到 `Model/output/pred_retrieve_merged.json` 文件中。

   用法範例：
   ```bash
   python generate_comparison.py bm25 grok-beta
   ```

### 目錄結構

- **Model**：
  - **bm25_words**：包含與 BM25 詞處理相關的文件，包括自定詞表、停用詞表和同義詞表，透過LLM及半人工方式生成。
  - **dataset**：包含初賽的題目和答案範例。
  - **output**：存放模型的輸出結果。
    - **pred_retrieve_merged.json**：最終合併的答案文件，包含不同檢索方法的合併結果。
  - **bm25.py**：BM25 檢索方法的實現。
  - **bm25_gpt.py**：使用 LLM 模型（如 gpt-4o 和 grok-beta）對 BM25 top n的檢索結果進行篩選，以提高檢索質量。
  - **generate_comparison.py**：用於生成不同檢索方法的比較報告，包括各方法的答案和合併答案。最終結果會輸出到 `Model/output/pred_retrieve_merged.json` 文件中。
- **Preprocess**：
  - **faq**：將 JSON 資料拆分為 TXT 文件，以便後續處理。
  - **finance**：
    - **pdf**：存放原始 PDF 文件。
    - **clean_gemini**：存放經 `cleanContent.py` 清洗後的文本內容。
    - **gemini_pro**：存放由 `extract_pdf_gemini.py` 提取的文本內容。
    - **cleanContent.py**：文本清洗腳本，用於進一步處理提取的文本內容。
    - **extract_pdf_gemini.py**：使用 Gemini 1.5 Pro 提取 PDF 文件中的文本內容的腳本。
    - **finance_mapping_sort.csv**：包含金融相關映射信息的 CSV 文件，透過 PDF 圖片比對完整財報加上半人工方式生成。
    - **pdf_image_comparison.ipynb**：Jupyter Notebook，用於對比處理過的 PDF 圖像。
  - **insurance**：
    - **1_input_pdfs**：存放原始 PDF 文件。
    - **clean_gemini**：存放經 `cleanContent.py` 清洗後的文本內容。
    - **gemini_pro**：存放由 `extract_pdf_gemini.py` 提取的文本內容。
    - **cleanContent.py**：文本清洗腳本，用於進一步處理提取的文本內容。
    - **extract_pdf_gemini.py**：使用 Gemini 1.5 Pro 提取 PDF 文件中的文本內容的腳本。
    - **find_issuance_mapping.py**：查找發行映射信息的腳本。
    - **insurance_mapping.csv**：包含保險相關映射信息的 CSV 文件，半人工生成。
- **.env**：存放 API 金鑰和其他環境變數的文件。
- **.env_temp**：`.env` 文件的模板，供用戶創建自己的配置。
- **README.md**：項目說明文件，包含使用說明和目錄結構。
- **requirements.txt**：項目所需的 Python 依賴項文件，Python採用 3.12.2。
- **shared_functions.py**：不同腳本中共享使用的實用函數。

### 配置

通過複製 `.env_temp` 創建 `.env` 文件，並根據您的環境更新必要的憑據和設置。

- `.env` 文件需要包含以下兩個 API 金鑰：
  - `GEMINI_API_KEY`：用於 `extract_pdf_gemini.py`。
  - `XAI_API_KEY`：用於 `bm25_gpt.py`。
