import argparse
import json
import os
import fnmatch
import re
from tqdm import tqdm
def find_all_py_files(dir):
    files = []
    for root, dirs, filenames in os.walk(dir):
        for filename in fnmatch.filter(filenames, '*.py'):
            files.append(os.path.join(root, filename))
    return files


def extract_mermaid(content):
    if '```mermaid' in content:
        p_code = re.compile(r'```mermaid\n(.*?)\n```', flags=re.DOTALL)
        code_block = p_code.findall(content)[0]
        return code_block
    else:
        return ""

def extract_code(content):
    if '```python' in content:
        p_code = re.compile(r'```python\n(.*?)\n```', flags=re.DOTALL)
        code_block = p_code.findall(content)[0]
        return code_block
    else:
        return ""
def agent(messages, api_config):
    from openai import OpenAI
    client = OpenAI(
        api_key=api_config["api_key"],
        base_url=api_config["base_url"],
    )
    response = client.chat.completions.create(
        model=api_config["model"],
        messages=messages,
        max_tokens=api_config["max_tokens"],
        temperature=api_config["temperature"],
        stop=api_config["stop"],
        n=api_config["n"],
    )
    responses = [response.choices[i].message.content for i in range(api_config["n"])]
    return responses, response.usage, response.model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--api_config", type=str, default="../configs/openai_api_key_config.json")
    parser.add_argument("--prompt_path", type=str, default="../../prompts/data_transform_en.txt")
    parser.add_argument("--data_dir", type=str, default="../../data/Math/easy")
    parser.add_argument("--output_dir", type=str, default="./output")
    args = parser.parse_args()
    print("-" * 20, "Args", "-" * 20)
    print(json.dumps(vars(args), indent=4))

    api_config = json.load(open(args.api_config))
    print("-" * 20, "API Config", "-" * 20)
    print(json.dumps(api_config, indent=4))

    prompt = open(args.prompt_path).read()
    print("-" * 20, "Prompt", "-" * 20)
    print(prompt)

    all_py_files = find_all_py_files(os.path.join(args.data_dir, "code"))
    # sort
    all_py_files = sorted(all_py_files)
    print("-" * 20, "Total Files", "-" * 20)
    print(len(all_py_files))
    print(all_py_files[:10])
    for file in tqdm(all_py_files):
        problem_path = file.replace("/code/", "/problems/").replace(".py", ".txt")
        problem = open(problem_path).read()
        save_path = file.replace("/code/", "/mermaid/").replace(".py", ".txt")
        if os.path.exists(save_path):
            continue
        solution = open(file).read()
        # my_solution = Solution()
        if "my_solution = Solution()" in solution:
            solution = solution.split("my_solution = Solution()")[0]
        tmp_prompt = prompt.replace("%%%problem%%%", problem)
        tmp_prompt = tmp_prompt.replace("%%%solution%%%", solution)
        messages = [
            {"role": "user", "content": tmp_prompt},
        ]
        print(json.dumps(messages, indent=4))
        responses, usage, model = agent(messages, api_config)
        mermaid = extract_mermaid(responses[0])
        open(save_path, "w").write(mermaid)
        test_cases = "\n\n" + extract_code(responses[0])
        open(file, "a").write(test_cases)







