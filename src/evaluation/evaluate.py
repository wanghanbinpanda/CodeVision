import re
import argparse
import json
import os
import fnmatch
from tqdm import tqdm
from openai import OpenAI
import anthropic
import os
import base64
import requests
import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer
from transformers import MllamaForConditionalGeneration, AutoProcessor
from transformers import AutoModelForCausalLM 


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

def extract_code(content):
    try:
        if '```python' in content:
            p_code = re.compile(r'```python\n(.*?)\n```', flags=re.DOTALL)
            code_block = p_code.findall(content)[0]
            if "assert" in code_block:
                code_block = code_block.split("assert")[0]
            return code_block
        elif '```' in content:
            p_code = re.compile(r'```(.*?)\n(.*?)```', flags=re.DOTALL)
            code_block = p_code.findall(content)[0][1]
            if "assert" in code_block:
                code_block = code_block.split("assert")[0]
            return code_block
        else:
            return content
    except:
        return content



#  base 64 编码格式
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def get_response(prompt,image_path,api_config):
    base64_image = encode_image(image_path)
    client = OpenAI(
        api_key=api_config["api_key"],
        base_url=api_config["base_url"],
    )
    response = client.chat.completions.create(
        model=api_config["model"],
        messages=[
            {
              "role": "user",
              "content": [
                {
                  "type": "image_url",
                  "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}"
                  }
                },
                {
                  "type": "text",
                  "text": prompt
                }
              ]
            }
          ],
        max_tokens = api_config["max_tokens"],
        temperature = api_config["temperature"],
        stop = api_config["stop"],
        top_p=api_config["top_p"],
        n = api_config["n"],
        )
    responses = [response.choices[i].message.content for i in range(api_config["n"])]
    return responses, response.usage, response.model

def get_response_claude(prompt,image_path,api_config):
    base64_image = encode_image(image_path)
    Baseurl = api_config["base_url"]
    Skey = api_config["api_key"]
    payload = json.dumps({
        "model": api_config["model"],
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_image}"
                        }
                    }
                ]
            }
        ],
        "max_tokens": api_config["max_tokens"],
        "temperature": api_config["temperature"],
        "stop": api_config["stop"],
        "top_p": api_config["top_p"],
        "n": api_config["n"]
    })
    url = Baseurl + "/v1/chat/completions"
    headers = {
        'Accept': 'application/json',
        'Authorization': f'Bearer {Skey}',
        'User-Agent': 'Apifox/1.0.0 (https://apifox.com)',
        'Content-Type': 'application/json'
    }

    response = requests.request("POST", url, headers=headers, data=payload)
    response = response.json()
    # print(response)
    responses = [response["choices"][i]["message"]["content"] for i in range(api_config["n"])]
    return responses, response["usage"], response["model"]

def get_response_internvl(prompt,image_path,api_config):
    import requests

    url = api_config["base_url"]  # （API）
    api_key = api_config["api_key"]  # （KEY）

    # example
    file_paths = [
        image_path
    ]
    question = prompt  # (Question)

    files = [('files', open(file_path, 'rb')) for file_path in file_paths]
    data = {
        'question': question,
        'api_key': api_key
    }

    responses = []
    try:
        response = requests.post(url, files=files, data=data)
        responses.append(response.json().get("response", "No response key found in the JSON."))
        if response.status_code == 200:
            pass
            # print("Response:", response.json().get("response", "No response key found in the JSON."))
        else:
            print("Error:", response.status_code, response.text)
    except requests.exceptions.RequestException as e:
        print(f"Error: {e}")
    return responses, response.json().get("usage", 0), response.json().get("model", "InternVL-2-Pro")

def get_response_minicpm(prompt,image_path,api_config,model,tokenizer):
    image = Image.open(image_path).convert('RGB')
    question = prompt
    msgs = [{'role': 'user', 'content': [image, question]}]

    res = model.chat(
        image=None,
        msgs=msgs,
        tokenizer=tokenizer
    )
    responses = [res]
    return responses, 0, api_config["model"]

def get_response_phi(prompt,image_path,api_config,model,processor):
    messages = [ 
        {"role": "user", "content": f"<|image_1|>\n{prompt}"}
    ] 
    image = Image.open(image_path)
    prompt = processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(prompt, [image], return_tensors="pt").to(model.device)
    generation_args = { 
        "max_new_tokens": api_config["max_tokens"], 
        "temperature": api_config["temperature"], 
        "do_sample": True, 
        "top_p":api_config["top_p"],
    } 
    generate_ids = model.generate(**inputs, eos_token_id=processor.tokenizer.eos_token_id, **generation_args) 
    # remove input tokens 
    generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
    response = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0] 
    responses = [response]
    return responses, 0, api_config["model"]

def get_response_llama(prompt,image_path,api_config,model,processor):
    image = Image.open(image_path)
    messages = [
        {"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": prompt}
        ]}
    ]
    input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(
        image,
        input_text,
        add_special_tokens=False,
        return_tensors="pt"
    ).to(model.device)
    generation_args = { 
        "max_new_tokens": api_config["max_tokens"], 
        "temperature": api_config["temperature"], 
        "do_sample": True, 
        "top_p":api_config["top_p"],
    } 
    generate_ids = model.generate(**inputs, eos_token_id=processor.tokenizer.eos_token_id, **generation_args)
    # remove input tokens 
    generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
    response = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0] 
    responses = [response]
    return responses, 0, api_config["model"]



def construct_prompt(starter_code):
    prompt = """Generate code according to flowchart.
Note: If you're using a specific python package, you'll need to import it yourself. You don't need to use functions like input() to get input, just complete the python function.
Starter Code:
```python
%%%starter_code%%%
```
Present the code between ```python and ```.
"""
    return prompt.replace("%%%starter_code%%%", starter_code)

def construct_prompt_mask(starter_code):
    prompt = """Generate code according to flowchart.
Note: If you're using a specific python package, you'll need to import it yourself. You don't need to use functions like input() to get input, just complete the python function.
Note: A small part of the flowchart may be masked, you need to understand the flowchart and then generate the complete code.
Starter Code:
```python
%%%starter_code%%%
```
Present the code between ```python and ```.
"""
    return prompt.replace("%%%starter_code%%%", starter_code)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--api_config", type=str, default="src/configs/openai_api_key_config.json")
    parser.add_argument("--data_path", type=str, default="data/HumanEval-V/HumanEval.jsonl")
    parser.add_argument("--image_dir", type=str, default="data/HumanEval-V/images")
    parser.add_argument("--output_dir", type=str, default="/output")
    args = parser.parse_args()
    print("-" * 20, "Args", "-" * 20)
    print(json.dumps(vars(args), indent=4))

    api_config = json.load(open(args.api_config))
    print("-" * 20, "API Config", "-" * 20)
    print(json.dumps(api_config, indent=4))

    problems = read_jsonl_file(args.data_path)
    print("Loading data from {}".format(args.data_path)," total tasks:", len(problems))
    ori_len = len(problems)
    samples = []

    dataset = args.data_path.split("/")[-2]
    args.output_dir = os.path.join(args.output_dir,dataset, api_config["model"])
    os.makedirs(args.output_dir, exist_ok=True)
    print("Update output dir to {}".format(args.output_dir))

    print("-" * 20, "Check existing results", "-" * 20)
    save_path = os.path.join(args.output_dir, "samples.jsonl")
    try:
        done_data = read_jsonl_file(save_path)
        done_ids = [x["task_id"] for x in done_data]
        print("Skipping {} tasks".format(len(done_ids)))
        problems = [x for x in problems if x["task_id"] not in done_ids]
        print("Remaining {} tasks".format(len(problems)))
        assert len(problems) + len(done_ids) == ori_len, "Skipping data and remaining tasks do not match the original data size"
    except Exception as e:
        print(e)
        print("No existing results!")
    print("-" * 20, "Starting", "-" * 20)


    if "minicpm" in args.api_config:
        # minicpm
        model_path = api_config["api_key"]
        model = AutoModel.from_pretrained(model_path, trust_remote_code=True,
            attn_implementation='sdpa', torch_dtype=torch.bfloat16) # sdpa or flash_attention_2, no eager
        model = model.eval().cuda()
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    elif "llama" in args.api_config:
        model_id = api_config["api_key"]
        model = MllamaForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        processor = AutoProcessor.from_pretrained(model_id)
    elif "phi_" in args.api_config:
        model_id = api_config["api_key"]
        model = AutoModelForCausalLM.from_pretrained(model_id, device_map="cuda", trust_remote_code=True, torch_dtype="auto", _attn_implementation='eager') # use _attn_implementation='eager' to disable flash attention
        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True) 

    with open(save_path, "a") as f:
        for i, problem in tqdm(enumerate(problems),desc="Generating samples",total=len(problems)):
            task_id = problem["task_id"]
            if "MASK" in args.image_dir:
                prompt = construct_prompt_mask(problem["starter_code"])
            else:
                prompt = construct_prompt(problem["starter_code"])
            # print(prompt)
            image_path = os.path.join(args.image_dir, task_id + ".png")
            if "internvl" in args.api_config:
                responses, usage, model_name = get_response_internvl(prompt, image_path, api_config)
            elif "claude" in args.api_config:
                responses, usage, model_name = get_response_claude(prompt, image_path, api_config)
            elif "minicpm" in args.api_config:
                responses, usage, model_name = get_response_minicpm(prompt, image_path, api_config,model,tokenizer)
            elif "llama" in args.api_config:
                responses, usage, model_name = get_response_llama(prompt, image_path, api_config,model,processor)
            elif "phi_" in args.api_config:
                responses, usage, model_name = get_response_phi(prompt, image_path, api_config,model,processor)
            else:
                responses, usage, model_name = get_response(prompt, image_path, api_config)
            problem["response"] = responses[0]
            problem["completion"] = extract_code(responses[0])
            problem["usage"] = str(usage)
            problem["model"] = str(model_name)
            f.write(
                json.dumps(problem) + "\n"
            )
            f.flush()  # make sure the output is written to file

    # execute bash
    cmd = f"evaluate_functional_correctness {save_path} --problem_file={args.data_path}"
    os.system(cmd)


    # calculate score
    results_file = save_path + "_results.jsonl"
    results = read_jsonl_file(results_file)
    from score import Score
    results_fine_grained = Score(results)
    print(results_fine_grained)
    result_save_path = save_path.replace("samples.jsonl", "results.txt")
    # write results as string to file
    with open(result_save_path, "a") as f:
        f.write(str(results_fine_grained))

    


