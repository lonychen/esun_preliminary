import re

# Define patterns and corresponding replacements
replacement_patterns = {
    r'一一二年': '112年',
    r'一一一年': '111年',
    r'一一○年': '110年',
    r'一月一日': '1月1日',
    r'七月一日': '7月1日',
    r'三月三十一日': '3月31日',
    r'九月三十日': '9月30日',
    r'2025\s*年': '114年',
    r'2024\s*年': '113年',
    r'2023\s*年': '112年',
    r'2022\s*年': '111年',
    r'2021\s*年': '110年',
    r'2020\s*年': '109年',
    r'2019\s*年': '108年',
    r'2018\s*年': '107年',
    r'2017\s*年': '106年',
    r'1999\s*年': '88年',
    r'1月1日至3月31日': '第1季',
    r'4月1日至6月30日': '第2季',
    r'7月1日至9月30日': '第3季',
    r'10月1日至12月31日': '第4季',
    r'1月1日至6月30日': '前2季',
    r'1月1日至9月30日': '前3季',
    r'前三季': '前3季',
    r'第一季': '第1季',
    r'第二季': '第2季',
    r'第三季': '第3季',
    r'1至3月': '第1季',
    r'1至9月': '前3季',
    r'7至9月': '第3季',
    r'營收表現': '綜合損益表中的營業收入',
    r'流動資產總額': '流動資產合計',
    r'新台幣？': '元？',
   
}

def filter_query(query):
    # Apply each pattern replacement
    for pattern, replacement in replacement_patterns.items():
        query = re.sub(pattern, replacement, query)
    return query
