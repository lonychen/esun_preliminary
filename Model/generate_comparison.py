import json
import os
import sys
import argparse
from typing import List

# 將上一層目錄加入到系統路徑中
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from shared_functions import filter_query

def load_answers(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return {str(item['qid']): item for item in data['ground_truths']}

def load_bm25_output(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    # Convert data to a dictionary with qid as the key
    return {str(item['qid']): item for item in data}

def load_sources(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return {str(item['qid']): item['source'] for item in data['questions']}

def generate_comparison_html(model_names: List[str], output_html="output/comparison.html",
                             bm25_output_path="output/bm25_retrieval_output.json",
                             sources_path="output/sources.json",
                             insurance_method="BM25", finance_method="BGE_Reranker", faq_method="GPT"):
    # Load BM25 output to get questions and document contents
    bm25_data = load_bm25_output(bm25_output_path)
    sources_data = load_sources(sources_path)
    
    # Load all answers and associate each method (model) with its JSON data
    method_answers = {}
    method_file_map = {}  # Map method name to the loaded data

    # Collect all methods used, including specified category methods
    all_methods = model_names + [insurance_method, finance_method, faq_method]
    all_methods = list(dict.fromkeys(all_methods))  # Remove duplicates while preserving order

    for model_name in all_methods:
        # Construct the file path
        file_path = os.path.join('output', f'truth_{model_name}.json')
        if not os.path.exists(file_path):
            print(f"Error: File not found for model '{model_name}' at path '{file_path}'")
            sys.exit(1)
        method_answers[model_name] = load_answers(file_path)
        method_file_map[model_name] = method_answers[model_name]  # Store the data under the method name

    # Use the first model as the "correct answers" source
    correct_method_name = model_names[0]
    correct_answers = method_answers[correct_method_name]

    # Associate each category with the specified retrieval method
    category_method_map = {
        "insurance": insurance_method,
        "finance": finance_method,
        "faq": faq_method
    }

    # Collect all qids and prepare comparison data
    all_qids = set()
    for answers in method_answers.values():
        all_qids.update(answers.keys())
    all_qids = sorted(all_qids, key=int)

    # Track correctness statistics
    correctness_stats = {method: 0 for method in all_methods}
    correctness_stats["Merged Answer"] = 0  # For the new "Merged Answer" method

    # Track category-specific correctness statistics
    categories = ["insurance", "finance", "faq"]
    category_correctness_stats = {category: {method: 0 for method in all_methods} for category in categories}
    for category in categories:
        category_correctness_stats[category]["Merged Answer"] = 0

    # Count total questions in each category
    category_totals = {category: 0 for category in categories}

    merged_answers = []

    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Method Comparison</title>
        <style>
            table {{ width: 100%; border-collapse: collapse; }}
            th, td {{ padding: 10px; text-align: center; border: 1px solid #ddd; }}
            th {{ background-color: #f2f2f2; }}
            .match {{ background-color: #d4edda; }}
            .mismatch {{ background-color: #f8d7da; }}
            .popup {{
                display: none;
                position: fixed;
                top: 5%;
                left: 5%;
                width: 90vw;
                height: 90vh;
                background-color: rgba(0, 0, 0, 0.8);
                color: #fff;
                overflow-y: auto;
                padding: 20px;
                z-index: 1000;
                border-radius: 10px;
            }}
            .popup-content {{
                position: relative;
                background-color: #333;
                padding: 15px;
                border-radius: 8px;
            }}
            .close-btn {{
                position: absolute;
                top: 10px;
                right: 15px;
                color: #fff;
                background: none;
                border: none;
                font-size: 18px;
                cursor: pointer;
            }}
            .popup .show {{
                display: block;
            }}
            .popup-overlay {{
                position: fixed;
                top: 0;
                left: 0;
                width: 100vw;
                height: 100vh;
                background: rgba(0, 0, 0, 0.5);
                z-index: 999;
            }}
        </style>
        <script>
            let currentPopup = null;

            function togglePopup(id) {{
                if (currentPopup) {{
                    document.getElementById(currentPopup).style.display = "none";
                }}
                var popup = document.getElementById(id);
                if (popup === currentPopup) {{
                    currentPopup = null;
                    return;
                }}
                popup.style.display = "block";
                currentPopup = id;
            }}

            function closePopup(id) {{
                document.getElementById(id).style.display = "none";
                currentPopup = null;
            }}
        </script>
        <style>
            .color-toggle {{
                background-color: #e7f4ff;
            }}
        </style>
        <script>
            function toggleRowColor(rowId) {{
                var row = document.getElementById(rowId);
                row.classList.toggle('color-toggle');
            }}
        </script>

    </head>
    <body>
        <h1>Comparison of Answers from Different Methods</h1>
        <p><strong>Correct Answer Source:</strong> {correct_method_name}</p>
        <table>
            <tr>
                <th>Toggle</th>  <!-- New header for toggle button -->
    """

    # Add method names as table headers in input order
    for method_name in all_methods:
        html_content += f"<th>{method_name}</th>"
    html_content += "<th>Merged Answer</th></tr>\n"

    # For each QID, add comparison rows
    for qid in all_qids:
        correct_answer_details = correct_answers.get(qid, {})
        correct_answer = correct_answer_details.get("retrieve", "N/A")
        category = correct_answer_details.get("category", "N/A")  # Change: Use correct category for each question

        if category in category_totals:
            category_totals[category] += 1

        chosen_method = category_method_map.get(category, correct_method_name)
        merged_answer = method_file_map[chosen_method].get(qid, {}).get("retrieve", "N/A")
        if merged_answer == -1:
            merged_answer = bm25_item.get("doc_ids", ["N/A"])[0]
        if isinstance(merged_answer, str):
            merged_answer = merged_answer.split('_')[0]

        bm25_item = bm25_data.get(qid, {})
        query = filter_query(bm25_item.get("query", "N/A"))
        doc_ids = bm25_item.get("doc_ids", [])

        doc_ids_cleaned = [doc_id.split('_')[0] for doc_id in doc_ids]
        doc_ids_display = ', '.join(doc_ids_cleaned)
        pdf_links = ""
        if category in ["insurance", "finance"]:  # Change: Correctly display PDF links for the relevant categories
            pdf_links = ' '.join([f"<a href='reference/{category}/{doc_id}.pdf' target='_blank'>{doc_id}</a>" for doc_id in doc_ids_cleaned])

        # Load source documents and create download links
        source_docs = sources_data.get(qid, [])
        sources_links = ' '.join([f"<a href='reference/{category}/{doc_id}.pdf' target='_blank'>{doc_id}</a>" for doc_id in source_docs])
  
        html_content += f"<td><strong>{qid}</strong><br>{pdf_links}<br>{query}<div class='possible_ids'>{sources_links}</div></td>"
  
        # Get reference answer for comparison
        correct_answer = correct_answers.get(qid, {}).get("retrieve", "N/A")
        category = correct_answers.get(qid, {}).get("category", "N/A")
        if category in category_totals:
            category_totals[category] += 1  # Count each question by category

        # Determine retrieval method based on category and retrieve merged answer
        chosen_method = category_method_map.get(category, correct_method_name)
        merged_answer = method_file_map[chosen_method].get(qid, {}).get("retrieve", "N/A")
        
        if merged_answer == -1:
            merged_answer = bm25_item.get("doc_ids", ["N/A"])[0]  # Defaults to the first document ID or "N/A" if no doc_ids

        # Clean `_part_x` suffix from merged_answer if it exists
        if isinstance(merged_answer, str):
            merged_answer = merged_answer.split('_')[0]  # Keep only the main part of the ID

        merged_answers.append({"qid": qid, "retrieve": merged_answer, "category": category})


        # Add answers with color coding and update correctness stats
        for method in all_methods:
            answers = method_answers[method]
            answer = answers.get(qid, {}).get("retrieve", "N/A")
            cell_class = "match" if str(answer) == str(correct_answer) else "mismatch"

            # Determine document ID with rank information and popup content
            answer_display = f"{answer} (N/A)"
            answer_popup_content = "N/A"
            if qid in bm25_data:
                doc_ids_list = bm25_data[qid].get("doc_ids", [])
                doc_texts = bm25_data[qid].get("doc_texts", [])
                for rank, (doc_id, doc_content) in enumerate(zip(doc_ids_list, doc_texts), start=1):
                    doc_id_main = doc_id.split('_')[0]
                    if str(doc_id_main) == str(answer):
                        answer_display = f"{answer} ({rank})"
                        answer_popup_content = doc_content.replace('\n', '<br>')
                        break

            # Add answer cell with rank information and popup
            html_content += f"""
            <td class='{cell_class}'>
                <div onclick="togglePopup('popup_{qid}_{method}')">{answer_display}</div>
                <div class="popup" id="popup_{qid}_{method}">
                    <div class="popup-content">
                        <button class="close-btn" onclick="closePopup('popup_{qid}_{method}')">&times;</button>
                        <p>{answer_popup_content}</p>
                    </div>
                </div>
            </td>
            """

            # Update correctness statistics
            if str(answer) == str(correct_answer):
                correctness_stats[method] += 1
                if category in category_correctness_stats:
                    category_correctness_stats[category][method] += 1

        # Add "Merged Answer" with color coding
        cell_class = "match" if str(merged_answer) == str(correct_answer) else "mismatch"

        # Get merged answer's document content
        merged_doc_text = ""
        if qid in bm25_data:
            doc_ids_list = bm25_data[qid].get("doc_ids", [])
            doc_texts = bm25_data[qid].get("doc_texts", [])
            for doc_id, doc_content in zip(doc_ids_list, doc_texts):
                doc_id_main = doc_id.split('_')[0]
                if str(doc_id_main) == str(merged_answer):
                    merged_doc_text = doc_content.replace('\n', '<br>')
                    break
        if not merged_doc_text:
            merged_doc_text = "N/A"

        html_content += f"""
        <td class='{cell_class}'>
            <div onclick="togglePopup('popup_{qid}_merged')">{merged_answer}</div>
            <div class="popup" id="popup_{qid}_merged">
                <div class="popup-content">
                    <button class="close-btn" onclick="closePopup('popup_{qid}_merged')">&times;</button>
                    <p>{merged_doc_text}</p>
                </div>
            </div>
        </td>
        """

        # Update correctness statistics for merged answer
        if str(merged_answer) == str(correct_answer):
            correctness_stats["Merged Answer"] += 1
            if category in category_correctness_stats:
                category_correctness_stats[category]["Merged Answer"] += 1

        html_content += "</tr>\n"

    # Add accuracy summary rows at the bottom of the table
    total_questions = len(all_qids)
    html_content += "<tr><td><strong>Total Correct</strong></td>"
    for method in all_methods + ["Merged Answer"]:
        html_content += f"<td>{correctness_stats[method]}</td>"
    html_content += "</tr>\n"

    html_content += "<tr><td><strong>Accuracy (%)</strong></td>"
    for method in all_methods + ["Merged Answer"]:
        accuracy = (correctness_stats[method] / total_questions) * 100
        html_content += f"<td>{accuracy:.2f}%</td>"
    html_content += "</tr>\n"

    # Add category-specific accuracy rows
    for category in categories:
        html_content += f"<tr><td><strong>{category.capitalize()} Accuracy (%)</strong></td>"
        for method in all_methods + ["Merged Answer"]:
            total_in_category = category_totals.get(category, 0)
            correct_in_category = category_correctness_stats[category][method]
            category_accuracy = (correct_in_category / total_in_category * 100) if total_in_category > 0 else 0.0
            html_content += f"<td>{category_accuracy:.2f}%</td>"
        html_content += "</tr>\n"

    # Closing HTML tags
    html_content += """
        </table>
    </body>
    </html>
    """


    # Save the HTML file
    os.makedirs(os.path.dirname(output_html), exist_ok=True)
    with open(output_html, 'w', encoding='utf-8') as file:
        file.write(html_content)

    print(f"Comparison HTML file generated: {output_html}")

    # Save merged answers to JSON files
    merged_output_path = "output/pred_retrieve_merged.json"
    merged_truth_output_path = "output/truth_merged.json"
    
    # Save merged answers
    os.makedirs(os.path.dirname(merged_output_path), exist_ok=True)
    with open(merged_output_path, 'w', encoding='utf-8') as file:
        json.dump({"answers": [{"qid": int(item["qid"]), "retrieve": str(item["retrieve"])} for item in merged_answers]}, file, indent=4, ensure_ascii=False)
    print(f"Merged answers JSON generated: {merged_output_path}")
    
    # Save merged answers with ground truths
    os.makedirs(os.path.dirname(merged_truth_output_path), exist_ok=True)
    with open(merged_truth_output_path, 'w', encoding='utf-8') as file:
        json.dump({"ground_truths": merged_answers}, file, indent=4, ensure_ascii=False)
    print(f"Merged answers ground truth JSON generated: {merged_truth_output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate comparison of answers with merged retrieval based on category.")
    parser.add_argument("model_names", nargs='+', help="Model names corresponding to truth_{model}.json files in the output directory.")
    parser.add_argument("--insurance_method", default="grok-beta", help="Method (model name) to use for the insurance category.")
    parser.add_argument("--finance_method", default="grok-beta", help="Method (model name) to use for the finance category.")
    parser.add_argument("--faq_method", default="grok-beta", help="Method (model name) to use for the FAQ category.")
    parser.add_argument("--bm25_output", default="output/bm25_retrieval_output.json", help="Path to the BM25 output file.")
    parser.add_argument("--sources_path", default="dataset/preliminary/questions_preliminary.json", help="Path to the sources file.")
    
    args = parser.parse_args()
    generate_comparison_html(args.model_names, bm25_output_path=args.bm25_output, sources_path=args.sources_path,
                             insurance_method=args.insurance_method, finance_method=args.finance_method,
                             faq_method=args.faq_method)