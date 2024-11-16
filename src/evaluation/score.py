
import re
import argparse
import json

def read_jsonl_file(file_path):
    results = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                results.append(json.loads(line.strip()))
            except:
                continue
    return results

def write_jsonl_file(file_path, data):
    with open(file_path, 'w', encoding='utf-8') as f:
        for line in data:
            f.write(json.dumps(line, ensure_ascii=False) + '\n')
def Score(results):
    """
    score function
    easy, medium, hard
    """
    keys = results[0].keys()
    res_len = len(results)
    easy, medium, hard = [], [], []
    if "meta" in keys:
        for line in results:
            if line["meta"]["difficulty"].lower() == "easy":
                easy.append(line)
            elif line["meta"]["difficulty"].lower() == "medium":
                medium.append(line)
            elif line["meta"]["difficulty"].lower() == "hard":
                hard.append(line)        
    elif "difficulty" in keys:
        for line in results:
            if line["difficulty"].lower() == "easy":
                easy.append(line)
            elif line["difficulty"].lower() == "medium":
                medium.append(line)
            elif line["difficulty"].lower() == "hard":
                hard.append(line)
    else:
        return ""

    easy_success = 0
    for line in easy:
        if line["passed"]:
            easy_success += 1
    easy_score = easy_success / len(easy)

    medium_success = 0
    for line in medium:
        if line["passed"]:
            medium_success += 1
    medium_score = medium_success / len(medium)

    hard_success = 0
    for line in hard:
        if line["passed"]:
            hard_success += 1
    hard_score = hard_success / len(hard)

    # 在一个字符串返回详细的计算过程
    return f"\n\nEasy: {easy_success}/{len(easy)} = {easy_score}\nMedium: {medium_success}/{len(medium)} = {medium_score}\nHard: {hard_success}/{len(hard)} = {hard_score}"
    

# main
if __name__ == "__main__":
    
    path = "/home/test/test05/whb/project/CodeVision/output/MATH/gpt-4o/samples.jsonl_results.jsonl"
    results = read_jsonl_file(path)
    print(Score(results))

    