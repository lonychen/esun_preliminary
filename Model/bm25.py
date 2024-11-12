import os
import json
import re
import importlib
import jieba
import time
import pandas as pd
import argparse
import sys
import math

# 將上一層目錄加入到系統路徑中
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from shared_functions import filter_query

sys.stdout.reconfigure(encoding='utf-8')

from tqdm import tqdm
from rank_bm25 import BM25Okapi  # 使用BM25算法进行文件检索

base_path = '.'

# 类别开关，设置为 True 时运行对应类别的检索
run_finance = True
run_insurance = True
run_faq = True
tokenizer_type='search' #（'standard'、'search'、'all'）
start_question = 0  # 可以更改這個值來選擇從哪一題開始
n = 3  # 您可以根據需要調整這個值
m = 2  # 您可以根據需要調整這個值

# 设置路径
dataset_path = os.path.join(base_path, 'dataset/preliminary')

# 文本路徑
extracted_finance_path = '../Preprocess/finance/clean_gemini'
extracted_insurance_path = '../Preprocess/insurance/clean_gemini'
extracted_faq_path = '../Preprocess/faq/faq_answer'

# 自定詞表路徑
dict_insurance_path = 'bm25_words/cust_dict_gemini_insurance.txt'
dict_insurance_path2 = 'bm25_words/cust_dict_my_insurance.txt'
dict_finance_path = 'bm25_words/cust_dict_gemini_finance.txt'
dict_finance_path2 = 'bm25_words/cust_dict_my_finance.txt'
dict_faq_path = 'bm25_words/cust_dict_gemini_faq.txt'
dict_faq_path2 = 'bm25_words/cust_dict_my_faq.txt'

# 斷字詞表路徑
stopwords_insurance_path = 'bm25_words/stopwords_insurance.txt'
stopwords_finance_path = 'bm25_words/stopwords_finance.txt'
stopwords_faq_path = 'bm25_words/stopwords_faq.txt'

# 同義詞表路徑
synonyms_insurance_path = 'bm25_words/synonyms_insurance.txt'
synonyms_finance_path = 'bm25_words/synonyms_finance.txt'
synonyms_faq_path = 'bm25_words/synonyms_faq.txt'

output_path = 'output/pred_retrieve_bm25.json'  # 输出文件的路径
output_truth_path = 'output/truth_bm25.json'  # 输出文件的路径

# 指定保存结果的 HTML 文件路径
results_file_path = 'output/best_params_results.html'

# 直接設置命令列參數解析
parser = argparse.ArgumentParser(description="BM25 and LLM retrieval script.")
parser.add_argument(
    "--question_path", 
    type=str, 
    default=".\\dataset\\preliminary\\questions_preliminary.json",
    help="Path to the question JSON file."
)
parser.add_argument(
    "--ground_truth_path", 
    type=str, 
    default=".\\dataset\\preliminary\\ground_truths_example.json",
    help="Path to the ground truth JSON file."
)
parser.add_argument(
    "--max_file_length",
    type=int,
    default=900,
    help="Maximum file length before splitting into chunks."
)
parser.add_argument(
    "--chunk_overlap",
    type=int,
    default=200,
    help="Number of characters of overlap between chunks."
)

args = parser.parse_args()
question_path = args.question_path
ground_truth_path = args.ground_truth_path
max_file_length = args.max_file_length
chunk_overlap = args.chunk_overlap

if chunk_overlap >= max_file_length:
    print("Error: chunk_overlap must be less than max_file_length.")
    sys.exit(1)
if chunk_overlap < 0:
    print("Error: chunk_overlap must be non-negative.")
    sys.exit(1)

# Finance: 文件與公司對應
finance_mapping_path = os.path.join(base_path, 'preprocess/finance/finance_mapping_sort.csv')  # 输出文件的路径

category_dict_paths = {
        'finance': [dict_finance_path, dict_finance_path2],
        #'finance':[dict_finance_path],
        #'finance':[],
        'insurance': [dict_insurance_path, dict_insurance_path2],
        'faq': [dict_faq_path, dict_faq_path2]
        #'faq': []
    }
# Initialize an empty DataFrame to store the results
df_columns = ['Question ID', 'Query', 'Ground Truth', 'BM25 Retrieved', 'BM25 Top3', 'LLM Retrieved', 'Result']
results_df = pd.DataFrame(columns=df_columns)

bm25_output_path = os.path.join(base_path, 'output/bm25_retrieval_output.json')
results_log_output_path = os.path.join(base_path, 'output/results_log.html')

# 定義公司名稱與股票代碼的對應表
company_code_mapping = {
    "2327": "國巨股份有限公司",          # 國巨
    "2357": "華碩電腦股份有限公司",      # 華碩
    "2308": "台達電子工業股份有限公司",  # 台達電
    "1216": "統一企業股份有限公司",      # 統一
    "2317": "鴻海精密工業股份有限公司",  # 鴻海
    "2395": "研華股份有限公司",          # 研華
    "2002": "中國鋼鐵股份有限公司",      # 中鋼
    "2207": "和泰汽車股份有限公司",      # 和泰
    "2303": "聯華電子股份有限公司",      # 聯電
    "1326": "台灣化學纖維股份有限公司",  # 台化
    "2412": "中華電信股份有限公司",      # 中華電信
    "2454": "聯發科技股份有限公司",      # 聯發科
    "2330": "台灣積體電路製造股份有限公司",# 台積電
    "2301": "光寶科技股份有限公司",      # 光寶科
    "2603": "長榮海運股份有限公司",      # 長榮
    "1590": "亞德客國際集團",            # 亞德客
    "2379": "瑞昱半導體股份有限公司",    # 瑞昱
    "1101": "台灣水泥股份有限公司",      # 台泥
    "2345": "智邦科技股份有限公司",   # 台泥
    "4587": "寶元數控股份有限公司"       # 寶元數控
}

#資產負債表
balance_sheet_items = [
    "現金及約當現金",
    "流動資產 透過按公允價值衡量之金融資產",
    "流動資產 按攤銷後成本衡量之金融資產",
    "流動資產 避險之金融資產",
    "應收票據",
    "關係人 應收票據淨額",
    "應收帳款",
    "關係人 應收帳款淨額",
    "其他應收款",
    "關係人 其他應收款",
    "所得稅資產",
    "存貨",
    "預付款項",
    "其他流動資產",
    "流動資產合計",
    
    "非流動資產 透過按公允價值衡量之金融資產",
    "非流動資產 按攤銷後成本衡量之金融資產",
    "採用權益法之投資",
    "不動產、廠房及設備",
    "使用權資產",
    "投資性不動產淨額",
    "無形資產",
    "遞延所得稅資產",
    "預付設備款",
    "存出保證金",
    "應收融資租賃款",
    "淨確定福利資產",
    "其他非流動資產",
    "非流動資產合計",
    
    "資產總額",
    "資產總計",    
    
    "短期借款",
    "應付短期票券",
    "流動負債 透過按公允價值衡量之金融負債",
    "流動負債 合約負債",
    "應付票據",
    "關係人 應付票據",
    "應付帳款",
    "關係人 應付帳款",
    "其他應付款",
    "本期所得稅負債",
    "流動負債 租賃負債",
    "一年內到期之長期負債",
    "其他流動負債",
    "流動負債合計",
    "流動負債總計",
    
    #非流動負債
    "非流動負債 合約負債",
    "應付公司債",
    "長期借款",
    "遞延所得稅負債",
    "非流動負債 租賃負債",
    "長期應付票據",
    "淨確定福利負債",
    "存入保證金",
    "其他非流動負債",
    "非流動負債合計",
    "非流動負債總計",
    
    "負債合計",
    "負債總計",
    
    # 歸屬於母公司業主之權益
    "普通股股本",
    "特別股股本",
    "資本公積",
    "保留盈餘",
    "法定盈餘公積",
    "特別盈餘公積",
    "未分配盈餘",
    "其他權益",
    
    "庫藏股票",
    "業主權益合計",
    "業主權益總計",
    "非控制權益",
    
    "權益總計",
    "負債及權益總計",
]

#綜合損益表
comprehensive_income_statement_items = [
    "營業收入",
    "營業成本",
    "營業毛利",
    
    "推銷費用",
    "管理費用",
    "研究發展費用",
    "營業費用合計",
    
    "營業淨利",
    "營業利益",
    
    "採用權益法認列之關聯企業及合資損益份額",
    "利息收入",
    "股利收入",
    "其他收入",
    "處份投資性不動產利益",
    "外幣兌換淨益",
    "透過損益按公允價值衡量之金融資產及負債之淨益(損)",
    "財務成本",
    "處份不動產、廠房及設備損失",
    "其他支出",
    "非金融資產減損損失",
    
    "營業外收入及支出合計",
    
    "稅前淨利",
    "所得稅費用",
    "本期淨利",
    "透過其他按公允價值衡量之權益工具投資未實現評價損失",
    "採用權益法認列之關聯企業及合資之其他綜合損益份額",
    
    "國外營運機構財報報表換算之兑換差額",
    
    "其他綜合損益",
    "本期綜合損益",
    "綜合損益總額",
    
    "淨利歸屬於母公司業主",
    "淨利歸屬於非控制權益",
    
    "綜合損益總額歸屬於母公司業主",
    "綜合損益總額歸屬於非控制權益",
    
    "基本每股盈餘",
    "稀釋每股盈餘",
]

#現金流量表
cash_flow_statement_items = [
    #營業活動之現金流量
    "稅前淨利",
    "折舊費用",
    "攤銷費用",
    #"按公允價值衡量之金融資產及負債之淨損(益)",
    
    "財務成本",
    "利息收入",
    "股利收入",
    "股份基礎給付酬勞成本",
    "處分不動產、廠房及設備淨損"
    "處分投資性不動產利益",
    "處分投資損失",
    "非金融資產減損損失",
    "存貨跌價損失",
    "存貨回升利益",
    "未實現外幣兌換淨損",
    
    "強制透過損益按公允價值衡量之金融資產",
    
    "應收票據",
    "應收帳款（增加）減少",
    "關係人 應付票據及帳款",
    "其他應收款",
    "關係人 其他應收款",
    "存貨（增加）減少",
    "預付款項",
    "其他流動資產",
    "合約負債",
    "應付票據及帳款",
    "其他應付款",
    "關係人 其他應付款",
    "其他流動負債",
    "淨確定福利負債",
    "營運產生之淨現金流入（流出）",
    "支付之所得稅",
    "營業活動之淨現金流入",
    
    #投資活動之現金流量
    "取得按公允價值衡量之金融資產",
    "透過按公允價值衡量之金融資產減資退回股款",
    "按攤銷後成本衡量之金融資產增加",
    "按攤銷後成本衡量之金融資產減少",
    "取得長期股權投資",
    "對子公司之收購",
    "取得不動產、廠房及設備",
    "處分不動產、廠房及設備價款",
    "取得無形資產",
    "取得投資性不動產",
    "處分投資性不動產價款",
    "應收融資租賃款減少",
    "其他非流動資產減少(增加)",
    "收取之利息",
    "收取之股利",
    "投資活動之淨現金流入（流出）",
    
    #籌資活動之現金流量
    "短期借款減少",
    "應付短期票券增加(減少)",
    "發行公司債",
    "舉借長期借款",
    "償還長期借款",
    "長期應付票券增加",
    "長期應付票券減少",
    "租賃本金償還",
    "其他非流動負債增加",
    "發放現金股利",
    "庫藏股轉讓員工",
    "庫藏股票買回成本",
    "取得子公司股權",
    "支付之利息",
    "非控制權益變動",
    "籌資活動之淨現金流入（流出）",
    
    "匯率變動對現金及約當現金之影響",
    "本期現金及約當現金增加（減少）",
    "期初現金及約當現金餘額",
    "期末現金及約當現金餘額",
]

#合併權益變動表
equity_change_statement_items = [
    #橫
    "普通股股本",
    "特別股股本",
    "資本公積",
    "法定盈餘公積",
    "特別盈餘公積",
    "未分配盈餘",
    "保留盈餘合計",
    
    "透過按公允價值衡量之金融資產未實現評價損益",
    "庫藏股票",
    "業主權益",
    "非控制權益",
    "合併權益總計",
    "合併權益總額",
    
    #直
    #"法定盈餘公積",
    #"普通股股東現金股利",
    #"特別股現金股利",
    #"普通股股票股利",
    
    "淨利",
    "本期淨利",
    #"其他綜合損益",
    
    "採用認列關聯企業及合資股權淨值之變動數",
    "對子公司所有權權益變動",
    "庫藏股轉讓員工認購之酬勞成本",
    "庫藏股轉讓員工",
    "庫藏股註銷",
    "處分已重估之資產迴轉特別盈餘公積",
   
    
]
    
# 读取停用词列表
def load_stopwords(stopwords_path):
    if not os.path.exists(stopwords_path):
        print(f"Warning: 停用词文件 {stopwords_path} 不存在，返回空的停用词集合。")
        return set()  # 返回空集合

    with open(stopwords_path, 'r', encoding='utf-8') as f:
        stopwords = set(f.read().splitlines())

    # 打印所有加载的停用词
    print(f"Loaded stopwords from {stopwords_path}:")
    #print(stopwords)

    return stopwords

# 判断一个词是否由数字和符号组成
def is_number_or_symbol(token):
    #return False
    return re.match(r'^[\d\W]+$', token) is not None

# 读取同義词列表
def load_synonyms(synonyms_path):
    synonyms_dict = {}
    if not os.path.exists(synonyms_path):
        print(f"Warning: 同義詞檔案 {synonyms_path} 不存在。")
        return synonyms_dict

    with open(synonyms_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                word, synonyms = line.split(' ', 1)
                synonyms_list = synonyms.split(', ')
                synonyms_dict[word] = synonyms_list

    return synonyms_dict

# 读取公司名称文件
def load_company_names(company_output_path):
    if os.path.exists(company_output_path):
        with open(company_output_path, 'r', encoding='utf8') as f:
            return json.load(f)
    else:
        print(f"Warning: Company names file {company_output_path} not found.")
        return {}

def load_finance_mapping(finance_mapping_path):
    if not os.path.exists(finance_mapping_path):
        print(f"Warning: Finance mapping file {finance_mapping_path} not found.")
        return {}

    # 讀取 CSV 並指定列名，確保 `pdf_name` 是字串
    df = pd.read_csv(finance_mapping_path, names=['PDF', 'Stock Code', 'File Path', 'Page', 'Page Count', 'DocType'], header=None)

    # 建立映射，從 PDF 名稱到股票代碼和文件路徑
    mapping_dict = {}
    for index, row in df.iterrows():
        pdf_name = row['PDF']  # `PDF` 現在是字串
        
        file_id = os.path.splitext(pdf_name)[0]  # 去除 '.pdf'

        # 處理可能的 NaN 值
        stock_code = str(int(row['Stock Code'])) if pd.notna(row['Stock Code']) else "0000"
        
        file_path_info = row['File Path']
        mapping_dict[file_id] = {'stock_code': stock_code, 'file_path': file_path_info, 'docType': row['DocType'] }

    return mapping_dict


# 根据股票代码获取公司名称
def get_company_name_by_code(stock_code, company_code_mapping):
    # 返回映射中的公司名称，如果找不到则返回 "未知公司"
    return company_code_mapping.get(stock_code, "未知公司")

# Load additional text for insurance files
def load_insurance_insertions(insurance_insertions_path):
    if not os.path.exists(insurance_insertions_path):
        print(f"Warning: Insurance insertions file {insurance_insertions_path} not found.")
        return {}

    df = pd.read_csv(insurance_insertions_path, header=None, names=["file_name", "insertion_text"])
    insertions_dict = {str(row["file_name"]): row["insertion_text"] for _, row in df.iterrows()}
    
    return insertions_dict

def load_data_from_text_with_insertion(source_path, insertion_dict, max_file_length=1000, chunk_overlap=0):
    corpus_dict = {}
    for file_name in tqdm(os.listdir(source_path), desc=f'Loading data from {source_path}'):
        if not file_name.endswith('.txt'):
            continue
        file_id = os.path.splitext(file_name)[0]  # Remove the '.txt' extension
        file_path = os.path.join(source_path, file_name)

        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
            insertion_text = insertion_dict.get(file_id, "Unknown document")  # Default text if no insertion is found
            
            # If the text is too long, split into chunks
            if len(text) > max_file_length:
                step_size = max_file_length - chunk_overlap
                if step_size <= 0:
                    print("Error: max_file_length must be greater than chunk_overlap.")
                    sys.exit(1)
                for i in range(0, len(text), step_size):
                    part_text = text[i:i + max_file_length]
                    text_with_finance_info = f"{insertion_text}\n{part_text}"
                    corpus_dict[f"{file_id}_part_{i // step_size + 1}"] = text_with_finance_info
            else:
                text_with_finance_info = f"{insertion_text}\n{text}"
                corpus_dict[file_id] = text_with_finance_info
    return corpus_dict

# 加载参考资料，返回一个字典，key为文件名，value为提取的文本内容
def load_data_from_text(source_path, process_text=False, max_file_length=1000, chunk_overlap=0):
    corpus_dict = {}
    for file_name in tqdm(os.listdir(source_path), desc=f'Loading data from {source_path}'):
        if not file_name.endswith('.txt'):
            continue
        file_id = file_name.replace('.txt', '')  # 保持file_id为字符串
        file_path = os.path.join(source_path, file_name)
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()

            if process_text:
                # 一行一行處理，去掉換行符，並對 "第xx條" 開頭的行進行重覆10次
                lines = text.splitlines()  # 逐行分割
                processed_lines = []
                for line in lines:
                    line = line.strip()  # 去掉前後的空白符
                    if line.startswith('第') and '條' in line:
                        processed_lines.append(line * 100)  # 重覆10次該行
                    else:
                        processed_lines.append(line)
                text = ''.join(processed_lines)  # 去掉所有換行符
            else:
                text = text.replace('\n', '').replace('\r', '')  # 僅去掉換行符

            # 如果文本长度超过 max_file_length，将文本分割成多个部分
            if len(text) > max_file_length:
                step_size = max_file_length - chunk_overlap
                if step_size <= 0:
                    print("Error: max_file_length must be greater than chunk_overlap.")
                    sys.exit(1)
                for i in range(0, len(text), step_size):
                    part_text = text[i:i + max_file_length]
                    corpus_dict[f"{file_id}_part_{i // step_size + 1}"] = part_text
            else:
                corpus_dict[file_id] = text
    return corpus_dict

def load_data_from_text_with_finance_info(
    source_path,
    finance_mapping,
    company_code_mapping,
    process_text=False,
    max_file_length=1000,
    chunk_overlap=0
):
    corpus_dict = {}
    sorted_files = sorted(os.listdir(source_path))

    for file_name in tqdm(sorted_files, desc=f'Loading data from {source_path}'):
        if not file_name.endswith('.txt'):
            continue
        file_id = file_name.replace('.txt', '')
        file_path = os.path.join(source_path, file_name)

        # Initialize part number for this file_id
        part_number = 1

        # Retrieve stock code and file path info from finance_mapping
        if file_id in finance_mapping:
            mapping_info = finance_mapping[file_id]
            stock_code = mapping_info.get('stock_code', "N/A")
            file_path_info = mapping_info.get('file_path', "")
            docType = mapping_info.get('docType', "")

            if not isinstance(docType, str):
                if isinstance(docType, float) and math.isnan(docType):
                    docType = ""
                else:
                    docType = str(docType)
                    
            # Process the file path info
            if isinstance(file_path_info, str):
                if '/' in file_path_info:
                    path_after_slash = file_path_info.split('/', 1)[1]
                else:
                    path_after_slash = file_path_info
                match = re.match(r'(\d{6})', path_after_slash)
                if match:
                    date_code = match.group(1)
                    year = int(date_code[:4]) - 1911
                    season = int(date_code[4:])
                    season_info = f"{year}年第{season}季"
                else:
                    season_info = "未知季度"
            else:
                season_info = "未知季度"
        else:
            stock_code = "N/A"
            season_info = "未知季度"
            docType = ""

        company_name = get_company_name_by_code(stock_code, company_code_mapping)

        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
            text = text.replace("合併財務報表", "")
            
            if process_text:
                text = text.replace('\n', '').replace('\r', '')

 
            if season_info != "未知季度" and '年' in season_info and '第' in season_info:
                year_season_match = re.match(r"(\d{3})年第(\d)季", season_info)
                if year_season_match and (docType == "合併現金流量表" or docType == "合併綜合損益表" or docType == "合併資產負債表" or docType == "合併權益變動表"):
                    year = int(year_season_match.group(1))
                    season = int(year_season_match.group(2))
                    text = text.replace("XX", f"{year}年第{season}季")
                    text = text.replace("YY", f"{year - 1}年第{season}季")
            
            # If text length exceeds max_file_length, split it into parts
            if len(text) > max_file_length:
                step_size = max_file_length - chunk_overlap
                if step_size <= 0:
                    print("Error: max_file_length must be greater than chunk_overlap.")
                    sys.exit(1)
                for i in range(0, len(text), step_size):
                    part_text = text[i:i + max_file_length]
                    text_with_finance_info = f"{stock_code} {company_name} {season_info} {docType} ({part_number})\n{part_text}"
                    corpus_dict[f"{file_id}_part_{part_number}"] = text_with_finance_info
                    part_number += 1
            else:
                text_with_finance_info = f"{stock_code} {company_name} {season_info} {docType} ({part_number})\n{text}"
                corpus_dict[f"{file_id}_part_{part_number}"] = text_with_finance_info
                part_number += 1

    return corpus_dict
    
def expand_query_with_synonyms(query_tokens, synonyms_dict):
    expanded_query = []

    for token in query_tokens:
        expanded_query.append(token)  # 保留原始詞彙
        if token in synonyms_dict:
            expanded_query.extend(synonyms_dict[token])  # 加入同義詞

    return expanded_query

def clear_dictionary():
    """ 清除自定義詞典 """
    jieba.dt.FREQ = {}  # 清空詞頻字典
    jieba.dt.total = 0  # 重置總詞頻計數
    jieba.dt.initialized = False  # 標記為未初始化，這樣在下次使用時會重新加載默認詞典
    #importlib.reload(jieba)
    print("Custom dictionaries have been cleared.")

def load_dictionary(paths):
    for path in paths:
        print(f"Loaded dictionary from {path}")
        jieba.load_userdict(path)
        try:
          if 'my' in path:

            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    word_info = line.strip().split()  # Split the line to extract the word
                    word = word_info[0]  # Extract the word itself
                    #print(f"Loaded cust word: {word}")  # Print the word

                    # Load the word into jieba without frequency or POS information
                    #jieba.add_word(word)
        except FileNotFoundError:
            print(f"Dictionary file {path} not found. Skipping.")


def switch_dictionary(category):
    """ 根據類型切換詞彙表 """

    if category in category_dict_paths:
        clear_dictionary()
        # 在清除後，確保初始化默認詞典
        if not jieba.dt.initialized:
            jieba.initialize()
        load_dictionary(category_dict_paths[category])
    else:
        raise ValueError(f"No dictionary available for category: {category}")

# 定義輸出檔案的函數
def export_retrieved_documents(output_file, retrieval_data):
    """
    匯出 BM25 檢索後的結果到 JSON 或 CSV 檔案
    :param output_file: 要儲存的檔案路徑 (e.g., 'retrieval_output.csv')
    :param retrieval_data: 包含題號、題目、文本等資訊的列表
    """
    df = pd.DataFrame(retrieval_data)
    if output_file.endswith('.csv'):
        df.to_csv(output_file, index=False, encoding='utf-8')
    elif output_file.endswith('.json'):
        df.to_json(output_file, orient='records', force_ascii=False, indent=4)
    else:
        print(f"Unsupported file format: {output_file}")

def get_docType(doc_info):
    docType = doc_info.get('docType', '')
    if not isinstance(docType, str):
        if isinstance(docType, float) and math.isnan(docType):
            docType = ''
        else:
            docType = str(docType)
    return docType.strip()



class BM25Retriever:
    def __init__(self, corpus_dict, stopwords, synonyms, tokenizer_type='search', pre_tokenized=False, finance_mapping=None):
        """
        Initialize BM25 Retriever.
        """
        self.corpus_dict = corpus_dict
        self.stopwords = stopwords
        self.synonyms = synonyms
        self.tokenizer_type = tokenizer_type
        self.pre_tokenized = pre_tokenized
        self.finance_mapping = finance_mapping
        self.tokenized_corpus = []
        self.corpus_list = []
        self.doc_id_list = []
        self.build_model()


    def tokenize(self, text):
        if self.pre_tokenized:
            tokens = [token for token in text.split() if token.strip() and token not in self.stopwords]
        else:
            if self.tokenizer_type == 'standard':
                # 标准分词
                tokens = [token for token in jieba.cut(text) if token.strip() and token not in self.stopwords and not is_number_or_symbol(token)]
            elif self.tokenizer_type == 'search':
                # 搜索引擎模式分词
                tokens = [token for token in jieba.cut_for_search(text)
                          if token.strip() and token not in self.stopwords and not is_number_or_symbol(token)]
            elif self.tokenizer_type == 'all':
                # 全模式分词
                tokens = [token for token in jieba.cut(text, cut_all=True)
                          if token.strip() and token not in self.stopwords and not is_number_or_symbol(token)]
            else:
                raise ValueError("Unknown tokenizer type")

        # 過濾掉只有一個字的分詞
        tokens = [token for token in tokens if len(token) > 1]

        # 打印分词结果
        #print(f"\nText: {text[:30]}... \n-> Tokens: {tokens}")

        return tokens

    def build_model(self):
        for doc_id, doc_text in tqdm(self.corpus_dict.items(), desc='Building BM25 model'):
            tokens = self.tokenize(doc_text)
            self.tokenized_corpus.append(tokens)
            self.corpus_list.append(doc_text)
            self.doc_id_list.append(str(doc_id))
        self.bm25 = BM25Okapi(self.tokenized_corpus)
    
    def retrieve(self, query, source, n=3, m=2, category=""):
        query = filter_query(query)
        
        # 提取指定来源的文档
        source_set = set(map(str, source))
        source_indices = [
            i for i, doc_id in enumerate(self.doc_id_list)
            if str(doc_id).split('_part')[0] in source_set
        ]
        if not source_indices:
            return None

        if category == 'finance' and self.finance_mapping:
            finance_keywords = ["資產負債表", "現金流量表", "綜合損益表", "權益變動表", "會計師"]

            # 从查询中提取财务关键字
            keywords_in_query = [keyword for keyword in finance_keywords if keyword in query]

            if keywords_in_query:
                filtered_source_indices = []
                for idx in source_indices:
                    doc_id = self.doc_id_list[idx]
                    original_doc_id = doc_id.split('_part')[0]
                    doc_info = self.finance_mapping.get(original_doc_id, {})
                    docType = get_docType(doc_info)

                    # 检查 docType 是否包含查询中的财务关键字
                    if any(keyword in docType for keyword in keywords_in_query):
                        filtered_source_indices.append(idx)

                # 如果过滤后有文档，更新 source_indices；否则保留原始 source_indices
                if filtered_source_indices:
                    source_indices = filtered_source_indices
                    
            
        # 将查询进行分词
        tokenized_query = self.tokenize(query)
        tokenized_query = expand_query_with_synonyms(tokenized_query, self.synonyms)

        # 只计算 source_indices 中文档的 BM25 分数
        scores = self.bm25.get_scores(tokenized_query)
        source_scores = [(i, scores[i]) for i in source_indices]
        source_scores.sort(key=lambda x: x[1], reverse=True)

        # 列印所有分頁及其分數
        #print(f"\nScores for query '{query}':")
        #for i, score in source_scores:
        #    print(f"Doc ID: {self.doc_id_list[i]}, Score: {score}")
        
            # 將分數結果按文檔分組
        doc_scores = {}
        for i, score in source_scores:
            doc_id = self.doc_id_list[i]
            original_doc_id = doc_id.split('_part')[0]
            if original_doc_id not in doc_scores:
                doc_scores[original_doc_id] = []
            doc_scores[original_doc_id].append((doc_id, score))

        # 對每個文檔的分頁按分數排序，並取前 m 個分頁
        for doc_id in doc_scores:
            doc_scores[doc_id].sort(key=lambda x: x[1], reverse=True)
            doc_scores[doc_id] = doc_scores[doc_id][:m]

        # 按文檔的最高分數排序，取前 n 個文檔
        sorted_docs = sorted(
            doc_scores.items(),
            key=lambda item: item[1][0][1],  # 文檔的最高分數
            reverse=True
        )[:n]

        # 收集結果
        top_n_indices = []
        for doc_id, parts in sorted_docs:
            top_n_indices.extend([doc_id for doc_id, _ in parts])
        
        if category == 'finance' and self.finance_mapping:
            # **新增的條件判斷**
            finance_keywords_numbers_1 = ["資產負債表", "現金流量表", "綜合損益表", "權益變動表"]
            finance_keywords_numbers_2 = ["會計師"]

            if any(word in query for word in comprehensive_income_statement_items + balance_sheet_items + cash_flow_statement_items + equity_change_statement_items):

                filtered_top_n_indices_1 = []
                filtered_top_n_indices_2 = []
                for doc_id in top_n_indices:
                    original_doc_id = doc_id.split('_part')[0]
                    doc_info = self.finance_mapping.get(original_doc_id, {})
                    docType = get_docType(doc_info)

                    # 檢查 docType 是否包含 finance_keywords_numbers 中的關鍵字
                    if any(keyword in docType for keyword in finance_keywords_numbers_1):
                        filtered_top_n_indices_1.append(doc_id)
                    if any(keyword in docType for keyword in finance_keywords_numbers_2):
                        filtered_top_n_indices_2.append(doc_id)

                # 如果過濾後有文檔，更新 top_n_indices；否則保留原始 top_n_indices
                if filtered_top_n_indices_1 and filtered_top_n_indices_2:
                    top_n_indices = filtered_top_n_indices_1
  
  
        return top_n_indices

# 读取停用词
stopwords_insurance = load_stopwords(stopwords_insurance_path)
stopwords_finance = load_stopwords(stopwords_finance_path)
stopwords_faq = load_stopwords(stopwords_faq_path)

# 读取同義詞
synonyms_insurance = load_synonyms(synonyms_insurance_path)
synonyms_finance = load_synonyms(synonyms_finance_path)
synonyms_faq = load_synonyms(synonyms_faq_path)

# 读取问题文件
with open(question_path, 'r', encoding='utf8') as f:
    qs_ref = json.load(f)

# 使用改写后的函数加载数据
finance_mapping = load_finance_mapping(finance_mapping_path)
#print(f"{finance_mapping}")


# 读取标准答案文件
try:
    with open(ground_truth_path, 'r', encoding='utf8') as f_gt:
        ground_truths = json.load(f_gt)
        ground_truths_dict = {item['qid']: item['retrieve'] for item in ground_truths['ground_truths']}
except FileNotFoundError:
    print(f"Warning: Ground truth file not found at {ground_truth_path}. Skipping ground truth comparisons.")
    ground_truths_dict = {}  # 空字典，避免后續比對錯誤

# Load insurance insertion text
insurance_insertions_path = os.path.join(base_path, 'preprocess\\insurance\\insurance_mapping.csv')
insurance_insertions_dict = load_insurance_insertions(insurance_insertions_path)

# 加载金融类参考资料并建立 BM25 模型
if run_finance:
    print("Building BM25 model for finance...")
    switch_dictionary("finance")
    corpus_dict_finance = load_data_from_text_with_finance_info(
        extracted_finance_path, finance_mapping, company_code_mapping, 
        max_file_length=max_file_length, chunk_overlap=chunk_overlap
    )
    bm25_finance = BM25Retriever(
        corpus_dict_finance, 
        stopwords_finance, 
        synonyms_finance, 
        tokenizer_type=tokenizer_type, 
        finance_mapping=finance_mapping  # Pass finance_mapping here
    )
# 加载保险类参考资料并建立 BM25 模型
if run_insurance:
    print("Building BM25 model for insurance...")
    switch_dictionary("insurance")
    corpus_dict_insurance = load_data_from_text_with_insertion(
        extracted_insurance_path, 
        insurance_insertions_dict,
        max_file_length=max_file_length, chunk_overlap=chunk_overlap
    )
    bm25_insurance = BM25Retriever(corpus_dict_insurance, stopwords_insurance, synonyms_insurance, tokenizer_type=tokenizer_type)

# 加载 FAQ 类参考资料并建立 BM25 模型
if run_faq:
    print("Building BM25 model for FAQ...")
    switch_dictionary("faq")
    corpus_dict_faq = load_data_from_text(
        extracted_faq_path, 
        max_file_length=max_file_length, chunk_overlap=chunk_overlap
    )
    bm25_faq = BM25Retriever(corpus_dict_faq, stopwords_faq, synonyms_faq, tokenizer_type=tokenizer_type)


# 初始化答案字典和计数器
answer_dict = {"answers": []}
truth_dict = {"ground_truths": []}
total_questions = 0
correct_answers_bm25 = 0
correct_answers_llm = 0
categories = ['insurance', 'finance', 'faq']
total_questions_per_category = {cat: 0 for cat in categories}
correct_answers_per_category = {cat: 0 for cat in categories}
retrieval_data = []  # 用來存儲檢索到的數據

# Group questions by category to avoid repeated resetting of jieba dictionary
questions_by_category = {
    'insurance': [],
    'finance': [],
    'faq': []
}

# Group questions by their category
for q_dict in qs_ref['questions']:
    category = q_dict['category']
    questions_by_category[category].append(q_dict)

# Process each category's questions in bulk, resetting dictionary only once per category
for category, questions in questions_by_category.items():
    if not questions:
        continue  # Skip if there are no questions in this category

    # Load the dictionary for the current category
    switch_dictionary(category)

    if category == 'finance' and run_finance:
        retriever = bm25_finance
    elif category == 'insurance' and run_insurance:
        retriever = bm25_insurance
    elif category == 'faq' and run_faq:
        retriever = bm25_faq
    else:
        continue  # Skip if the category is not enabled

    # Process all questions for the current category
    for q_dict in tqdm(questions, desc=f'Processing {category} questions'):
        qid = q_dict['qid']
        source = q_dict['source']
        query = q_dict['query']

        if int(qid) < start_question:
            continue
            
#        print(f"\n題號: {qid}")
        total_questions += 1
        total_questions_per_category[category] += 1

        # 設定 FAQ 類別的 n 值為 5
        category_n = n if category == 'faq' else n
        category_m = m if category == 'finance' else 1

        # 先執行 BM25 檢索，將 category_n 傳入作為參數
        retrieved_list = retriever.retrieve(query, source, n=category_n, m=category_m, category=category)

        if not retrieved_list:
            print(f"\nNo documents retrieved for question ID: {qid}")
            continue

        # 儲存檢索到的文本及其相關資訊
        doc_info = {
            "qid": qid,
            "query": query,
            "doc_ids": retrieved_list,
            "doc_texts": [retriever.corpus_dict[doc_id] for doc_id in retrieved_list],
        }
        retrieval_data.append(doc_info)

        # 執行關鍵字過濾邏輯
        tokenized_query = retriever.tokenize(query)

        most_relevant_doc_id = retrieved_list[0]
        most_relevant_doc_id = int(most_relevant_doc_id.split('_part')[0])
        ground_truth = ground_truths_dict.get(qid)

        # Compare results with ground truth
        bm25_correct = (ground_truth is not None and ground_truth == int(retrieved_list[0].split('_part')[0]))

        tokenized_query = retriever.tokenize(query)
        tokenized_query = expand_query_with_synonyms(tokenized_query, retriever.synonyms)

#        if not bm25_correct:
#          print(f"\n題號: {qid}")
#          print(f"題目: {query}")
#          print(f"題目分詞結果: {tokenized_query}")
#          print(f"正確答案ID: {ground_truth}")
#          print(f"BM25的前{n}個文本ID: {', '.join(retrieved_list)}")

          # 為前三個文檔列出前30個分詞
#          for doc_id in retrieved_list[:n]:
#            doc_text = retriever.corpus_dict.get(doc_id, "")
#            first_30_tokens = retriever.tokenize(doc_text)[:50]
#            print(f"文本ID {doc_id} 的前30個分詞: {first_30_tokens}")

        answer_dict['answers'].append({
            "qid": qid,
            "retrieve": most_relevant_doc_id
        })
        
        truth_dict['ground_truths'].append({
            "qid": qid,
            "retrieve": most_relevant_doc_id,
            "category": category
        })

        if bm25_correct:
            correct_answers_bm25 += 1
            correct_answers_per_category[category] += 1

# 將檢索結果匯出到檔案（指定 CSV 或 JSON）
export_retrieved_documents(bm25_output_path, retrieval_data)

# Save final answer JSON
with open(output_path, 'w', encoding='utf8') as f:
    json.dump(answer_dict, f, ensure_ascii=False, indent=4)
with open(output_truth_path, 'w', encoding='utf8') as f:
    json.dump(truth_dict, f, ensure_ascii=False, indent=4)

def append_results_to_html(total_questions, accuracy_bm25, accuracy_llm, categories, total_questions_per_category, correct_answers_per_category, max_file_length, chunk_overlap):
    """
    将每次运行结果追加为指定的 HTML 文件中的表格行。
    """
    # HTML 表格行内容
    table_row = f"""
    <tr>
        <td>{max_file_length}</td>
        <td>{chunk_overlap}</td>
        <td>{total_questions}</td>
        <td>{accuracy_bm25:.2f}%</td>
        <td>{accuracy_llm:.2f}%</td>
    """
    
    # 每个类别的统计结果
    for category in categories:
        total = total_questions_per_category[category]
        correct = correct_answers_per_category[category]
        accuracy = (correct / total * 100) if total > 0 else 0.0
        table_row += f"<td>{total}</td><td>{correct}</td><td>{accuracy:.2f}%</td>"

    table_row += "</tr>\n"
    
    # 如果文件不存在，则创建新文件并写入表头
    if not os.path.exists(results_file_path):
        with open(results_file_path, 'w', encoding='utf-8') as f:
            # 表格表头
            f.write("<html><head><meta charset='utf-8'><title>BM25 Parameters Results</title></head><body>")
            f.write("<h1>BM25 Retrieval Results</h1><table border='1' cellpadding='5' cellspacing='0' style='border-collapse:collapse; width:100%;'>")
            f.write("<tr><th>Max File Length</th><th>Chunk Overlap</th><th>Total Questions</th><th>BM25 Accuracy</th><th>LLM Accuracy</th>")
            for category in categories:
                f.write(f"<th>{category} Total</th><th>{category} Correct</th><th>{category} Accuracy</th>")
            f.write("</tr>")  # 关闭表头行

    # 将当前运行结果追加为表格的一行
    with open(results_file_path, 'a', encoding='utf-8') as f:
        f.write(table_row)

# 在所有运行完成后，仅添加关闭标签
def close_html_table():
    with open(results_file_path, 'a', encoding='utf-8') as f:
        f.write("</table></body></html>")


# 计算并打印总体结果
accuracy_bm25 = correct_answers_bm25 / total_questions * 100

# 打印到控制台
print(f"Total questions: {total_questions}")
print(f"BM25 Accuracy: {accuracy_bm25:.2f}%")

for category in categories:
    total = total_questions_per_category[category]
    correct = correct_answers_per_category[category]
    acc = correct / total * 100 if total > 0 else 0
    print(f"Category: {category}")
    print(f"  Total questions: {total}")
    print(f"  Correct answers: {correct}")
    print(f"  Accuracy: {acc:.2f}%")


