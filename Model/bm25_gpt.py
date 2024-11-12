import os
import json
import argparse
import sys
import time
import re

# 將上一層目錄加入到系統路徑中
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from shared_functions import filter_query
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI

# Load environment variables from .env file
load_dotenv()

# Paths setup
base_path = '.'
bm25_output_path = os.path.join(base_path, 'output/bm25_retrieval_output.json')
bm25_truth_path = os.path.join(base_path, 'output/truth_bm25.json')

def get_output_paths(model_name):
    """根據模型名稱動態設置輸出路徑。
    
    Args:
        model_name (str): 模型名稱，用於設置輸出文件路徑。
    
    Returns:
        tuple: 包含 OpenAI 輸出路徑和標準答案輸出路徑的元組。
    """
    openai_output_path = os.path.join(base_path, f'output/pred_retrieve_{model_name}.json')
    openai_truth_output_path = os.path.join(base_path, f'output/truth_{model_name}.json')
    return openai_output_path, openai_truth_output_path

def initialize_llm(model_name):
    """根據模型名稱初始化 LLM 模型。
    
    Args:
        model_name (str): 模型名稱。
    
    Returns:
        ChatOpenAI: 已初始化的 LLM 模型。
    """
    if model_name == "grok-beta":
        llm = ChatOpenAI(
            model=model_name,
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=2,
            api_key=os.getenv("XAI_API_KEY"),
            base_url="https://api.x.ai/v1"
        )
    else:
        llm = ChatOpenAI(
            model=model_name,
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=2
        )
    return llm
  
def load_existing_answers(output_path):
    """從文件中載入現有的答案（如果文件存在）。
    
    Args:
        output_path (str): 輸出文件的路徑。
    
    Returns:
        dict: 已存在的答案字典，如果文件不存在，則返回空字典。
    """
    if os.path.exists(output_path):
        with open(output_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}  # Return an empty dictionary instead of an empty list

def save_updated_answers(output_path, answers):
    """將更新後的答案保存回文件中。
    
    Args:
        output_path (str): 輸出文件的路徑。
        answers (dict): 要儲存的答案字典。
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(answers, f, ensure_ascii=False, indent=4)

def find_most_relevant_doc(query, documents, qid, retrieved_list, llm, retries=2):
    """使用 OpenAI Chat 模型找到與查詢最相關的文件。
    
    Args:
        query (str): 查詢字串。
        documents (list): 文件列表。
        qid (int): 問題 ID。
        retrieved_list (list): 檢索到的文件 ID 列表。
        llm (ChatOpenAI): LLM 模型實例。
        retries (int): 重試次數。

    Returns:
        int: 最相關文件的索引，若無相關文件則返回 -1。
    """
    print(f"\nProcessing QID {qid} with query: '{query}'")
    print(f"Number of documents to evaluate: {len(documents)}")

    prompt = (
        f"You are tasked with identifying the most relevant document that answers the following question:\n"
        f"Question: \"{query}\"\n\n"
        f"Below is a list of documents. Each document is labeled with a number:\n"
    )

    for idx, doc in enumerate(documents):
        prompt += f"Document {idx + 1}:\n{doc}\n\n"

    #prompt += "Please respond with only the number of the document (e.g., 1, 2, 3, ...), which you think best addresses the question."
    prompt += (
        "Please respond with only the number of the document (e.g., 1, 2, 3, ...), which you think best addresses the question. "
        "If you believe none of the documents are relevant or correct, respond with -1."
    )

    try:
        response = llm.invoke(prompt)
        response_content = response.content.strip()
        print(f"Model response: {response_content}")

        if response_content == "-1":
            print(f"QID: {qid}, No relevant document found. Model response was -1.")
            return -1  # Indicate no relevant document found
        elif response_content.isdigit():
            most_relevant_index = int(response_content) - 1
        elif '_part_' in response_content:
            most_relevant_index = int(response_content.split('_')[0]) - 1
        else:
            print("Response format is unrecognized.")
            most_relevant_index = 0  # Default to the first document

        if 0 <= most_relevant_index < len(retrieved_list):
            selected_doc_id = retrieved_list[most_relevant_index]
            print(f"QID: {qid}, Selected Document Number: {most_relevant_index + 1}, Doc ID: {selected_doc_id}")
        else:
            most_relevant_index = 0
            print(f"QID: {qid}, Selected Document Number: Invalid (out of range)")

        return most_relevant_index

    except Exception as e:
        print(f"Error in processing QID {qid} with LLM: {e}. Retrying in 60 seconds...")
        if retries > 0:
            time.sleep(60)
            return find_most_relevant_doc(query, documents, qid, retrieved_list, llm, retries - 1)
        else:
            print("Exhausted retries. Exiting.")
            sys.exit(1)

def find_most_relevant_doc_delete_method(query, documents, qid, retrieved_list, llm, retries=2):
    """使用 OpenAI Chat 模型逐步刪除最不相關的文件，最終找到最相關的文件。
    
    Args:
        query (str): 查詢字串。
        documents (list): 文件列表。
        qid (int): 問題 ID。
        retrieved_list (list): 檢索到的文件 ID 列表。
        llm (ChatOpenAI): LLM 模型實例。
        retries (int): 重試次數。
    
    Returns:
        int: 最後剩餘文件的索引。
    """
    print(f"\nProcessing QID {qid} with query: '{query}'")
    

    while len(documents) > 1:
        print(f"Number of documents to evaluate: {len(documents)}")
        
        prompt = (
            f"You are tasked with identifying and removing the least relevant document "
            f"for the following question:\nQuestion: \"{query}\"\n\n"
            f"Below is a list of documents. Each document is labeled with a number:\n"
        )

        for idx, doc in enumerate(documents):
            prompt += f"Document {idx + 1}:\n{doc}\n\n"

        prompt += (
            "Please respond with only the number of the document (e.g., 1, 2, 3, ...) "
            "you think is the least relevant to the question. If all documents are highly relevant and it's difficult to decide, "
            "simply respond with the number of the last document in the list."
        )

        try:
            # Call Gemini model
            response = llm.invoke(prompt)
            print(f"Received response from OpenAI for QID {qid}.")
            response_content = response.content.strip()  # Use .content directly
            print(f"Model response: {response_content}")

            # 解析模型回覆，將回覆中的數字轉成文件索引
            if response_content.isdigit():
                least_relevant_index = int(response_content) - 1  # Convert to zero-based index
                
                # 刪除最不相關的文件
                if 0 <= least_relevant_index < len(documents):
                    removed_doc_id = retrieved_list.pop(least_relevant_index)
                    removed_document = documents.pop(least_relevant_index)
                    print(f"Removed Document Number: {least_relevant_index + 1}, Doc ID: {removed_doc_id}")
                else:
                    print("Response index out of range. Skipping removal.")
            else:
                print("Response format is unrecognized. Skipping removal.")

        except Exception as e:
            print(f"Error in processing QID {qid} with LLM: {e}. Retrying in 60 seconds...")
            if retries > 0:
                print(f"Retrying... ({retries - 1} retries left)")
                time.sleep(60)
                return find_most_relevant_doc_delete_method(query, documents, qid, retrieved_list, retries - 1)
            else:
                print("Exhausted retries. Exiting.")
                sys.exit(1)
                
    # 最後僅剩下一個文件，返回其索引（檢查是否在 retrieved_list 中的索引位置）
    return 0


def process_retrieval_output(start_qid, model_name):
    """處理檢索輸出並從 QID 開始檢索最相關的文件。
    
    Args:
        start_qid (int): 開始的問題 ID。
        model_name (str): 要使用的模型名稱。
    """
    print(f"Starting retrieval output processing from QID {start_qid} using model {model_name}...")
    openai_output_path, openai_truth_output_path = get_output_paths(model_name)
    llm = initialize_llm(model_name)

    existing_openai_answers = load_existing_answers(openai_output_path).get("answers", [])
    existing_openai_truths = load_existing_answers(openai_truth_output_path).get("ground_truths", [])

    # Load BM25 truths from bm25_truth_path
    bm25_truths_data = load_existing_answers(bm25_truth_path)
    bm25_truths = bm25_truths_data.get("ground_truths", [])
    bm25_truths_dict = {item["qid"]: item for item in bm25_truths}

    # Load BM25 output from bm25_output_path
    bm25_output = load_existing_answers(bm25_output_path)
    # If bm25_output is a dict with a key "results", extract it
    if isinstance(bm25_output, dict):
        bm25_output = bm25_output.get("results", [])
    elif not isinstance(bm25_output, list):
        print("Error: bm25_output should be a list of dictionaries.")
        print(f"Actual content of bm25_output: {bm25_output}")
        return

    openai_answers_dict = {item["qid"]: item for item in existing_openai_answers}
    openai_truths_dict = {item["qid"]: item for item in existing_openai_truths}

    for item in bm25_output:
        if not isinstance(item, dict):
            print(f"Warning: Expected each item to be a dictionary, but found {type(item)}. Skipping this entry.")
            continue

        qid = item.get('qid')
        if qid is None or qid < start_qid:
            continue

        query = filter_query(item.get('query', ""))
        doc_texts = item.get('doc_texts', [])
        retrieved_list = item.get('doc_ids', [])

        if not query or not doc_texts or not retrieved_list:
            print(f"Warning: QID {qid} has missing 'query', 'doc_texts', or 'doc_ids'. Skipping this entry.")
            continue

        category = bm25_truths_dict.get(qid, {}).get("category", "")

        #most_relevant_index = find_most_relevant_doc_delete_method(query, doc_texts, qid, retrieved_list, llm)
        most_relevant_index = find_most_relevant_doc(query, doc_texts, qid, retrieved_list, llm)

        try:
            most_relevant_index = int(most_relevant_index)
        except ValueError:
            print(f"Error: most_relevant_index '{most_relevant_index}' is not an integer.")
            most_relevant_index = -1
        
        if most_relevant_index is not None and 0 <= most_relevant_index < len(retrieved_list):
            most_relevant_doc_id = retrieved_list[most_relevant_index]
            clean_doc_id = int(most_relevant_doc_id.split('_')[0])
            openai_answers_dict[qid] = {"qid": qid, "retrieve": clean_doc_id}
            openai_truths_dict[qid] = {"qid": qid, "retrieve": clean_doc_id, "category": category}
            print(f"Updated answers for QID {qid} with Document ID {clean_doc_id}.")
        else:
            print(f"Could not find a valid most relevant document for QID {qid}.")
            openai_answers_dict[qid] = {"qid": qid, "retrieve": -1}
            openai_truths_dict[qid] = {"qid": qid, "retrieve": -1, "category": category}

        updated_openai_answers = list(openai_answers_dict.values())
        updated_openai_truths = list(openai_truths_dict.values())
        save_updated_answers(openai_output_path, {"answers": updated_openai_answers})
        save_updated_answers(openai_truth_output_path, {"ground_truths": updated_openai_truths})
        print(f"Saved results for QID {qid}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Retrieve most relevant documents using OpenAI.")
    parser.add_argument("--start_qid", type=int, default=1, help="Starting question ID for calling LLM")
    parser.add_argument("--model", type=str, default="gpt-4o", help="Model to use, e.g., gpt-4o, grok-beta")
    args = parser.parse_args()
    process_retrieval_output(args.start_qid, args.model)
