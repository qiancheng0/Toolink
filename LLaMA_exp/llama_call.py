import math
from transformers import LlamaTokenizer, LlamaForCausalLM
import re
from tqdm import tqdm
import json
from utils import process_code
    

toolkits = json.load(open("../toolkits.json", "r"))

for test in ["date", "matrix", "arithmetic", "orientation", "remainder", "dyck", "track_shuffle", "boolean"]:
    # ======================================================================== #
    # Please set it to False when applying llama-toolink, se to True otherwise
    want_prompt = False
    if want_prompt:
        f = open(f"../prompt_lib/prompt_tool_llama/{test}_call.txt", "r")
        prompt = f.read().strip()
        f.close()
    
    # Choose from alpaca-raw, llama-raw, and llama-toolink
    model = "llama-toolink"
    test_file = f"{test}_{model}"
    code_file = "../code_exec/tmp0"
    save_path = f"../results_LLaMA/test_tool_call/{test_file}.md"
    
    if model == "alpaca-raw":
        model_path = "{PATH_TO_ALPACA}"
    if model == "llama-raw":
        model_path = "../LLaMA_Train/LLaMA-7B"
    if model == "llama-toolink":
        model_path = "../LLaMA_Train/LLaMA-Toolink"
    # ======================================================================== #

    f = open(f"../datasets/test_tool/{test}_testcall.jsonl", "r")
    lines = f.readlines()
    f.close()
    
    toolkit = toolkits[test]
    all_codes = toolkit[1]
    
    all_qst = []
    all_ans = []
    all_res = []

    for line in tqdm(lines):
        qst = ""
        line = json.loads(line)
        if want_prompt:
            qst += prompt + "\n\n"
        
        qst += line["prompt"]
        
        all_qst.append(qst)
        all_ans.append(str(line["answer"]))
        all_res.append(line["response"].strip())

    tokenizer = LlamaTokenizer.from_pretrained(model_path, padding_side="left")
    model = LlamaForCausalLM.from_pretrained(model_path)
    model.to("cuda")
    if tokenizer.pad_token_id == None:
        tokenizer.add_special_tokens({"bos_token": "<s>", "eos_token": "</s>", "pad_token": "<pad>"})
    model.resize_token_embeddings(len(tokenizer))

    if want_prompt:
        f = open(save_path, "w")
        f.write(prompt + "\n\n==============================prompt above! begin now!=====================================\n\n")
        f.close()
    else:
        f = open(save_path, "w")
        f.write("==============================no prompt! begin now!=====================================\n\n")
        f.close()
            
    bsz = 12
    st = 0
    bad_exec = 0
    no_code = 0
    bad_ans = 0
    good_ans = 0

    for _ in range(math.ceil(len(all_qst)/bsz)):
        if st + bsz >= len(all_qst):
            cur_qst = all_qst[st:]
            cur_ans = all_ans[st:]
            cur_res = all_res[st:]
        else:
            cur_qst = all_qst[st:st+bsz]
            cur_ans = all_ans[st:st+bsz]
            cur_res = all_res[st:st+bsz]
        
        st += bsz
        inputs = tokenizer(
            cur_qst,
            padding=True,
            return_tensors="pt"
        )

        generated_outputs = model.generate(
            input_ids=inputs["input_ids"].cuda(),
            attention_mask=inputs["attention_mask"].cuda(),
            max_new_tokens=256,
            eos_token_id=tokenizer.eos_token_id,
            bos_token_id=tokenizer.bos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
        decoded_output = tokenizer.batch_decode(
            generated_outputs, skip_special_tokens=True)
        
        cur_idx = -1
        for out in decoded_output:
            cur_idx += 1
            res:str = out.strip()
            
            if want_prompt:
                try:
                    prompt_tail = prompt[-40:].replace("</s>", "").strip()
                    res = res[res.index(prompt_tail)+len(prompt_tail):]
                    res = "### Instruction:\n" + res.split("### Instruction:")[1].split("### Instruction:")[0].strip()
                except:
                    print("Error in prompt parsing")
            
            try:
                f = open(save_path, "a")
                f.write(res + "\n\n")
                f.close()
            except:
                f = open(save_path, "a")
                f.write("Sorry, error in encoding")
                f.close()
            
            response_st = res.index("### Response:") + 13
            res = res[response_st:].split("### Input")[0].split("### Instruction")[0].strip()
            sim_code = all_codes
            
            success_exec, info = process_code(sim_code + "\n\n" + res, code_file)     
            if success_exec:
                print("~~~ GPT exec good ~~~")
                ans_found = False
                correct_ans = cur_ans[cur_idx]
                
                if test in ["arithmetic", "remainder"]:
                    ans_num = re.findall(r'-?\d+\.?\d*', info)
                    ans_num = [float(x) for x in ans_num]
                    for ans in ans_num:
                        if round(ans, 2) == round(float(correct_ans), 2):
                            f = open(save_path, "a")
                            f.write("\n" + info + "\n")
                            f.write(f"\ncorrect answer!\nThe correct ans is {correct_ans}\n")
                            f.close()
                            good_ans += 1
                            ans_found = True
                            break
                elif test in ["date", "orientation"]:
                    if correct_ans in info:
                        f = open(save_path, "a")
                        f.write("\n" + info + "\n")
                        f.write(f"\ncorrect answer!\nThe correct ans is {correct_ans}\n")
                        f.close()
                        good_ans += 1
                        ans_found = True
                elif test in ["dyck", "track_shuffle", "boolean"]:
                    if info.strip() in correct_ans.strip():
                        f = open(save_path, "a")
                        f.write("\n" + info + "\n")
                        f.write(f"\ncorrect answer!\nThe correct ans is {correct_ans}\n")
                        f.close()
                        good_ans += 1
                        ans_found = True
                elif test in ["matrix"]:
                    ans_lists = re.findall(r"\[(.*?)\]", info, re.S)
                    for ans in ans_lists:
                        try:
                            ans = eval("[" + ans + "]")
                            if ans == eval(correct_ans):
                                f = open(save_path, "a")
                                f.write("\n" + info + "\n")
                                f.write(f"\ncorrect answer!\nThe correct ans is {correct_ans}\n")
                                f.close()
                                good_ans += 1
                                ans_found = True
                                break
                        except:
                            pass
                    
                if not ans_found:
                    f = open(save_path, "a")
                    f.write("\n" + info + "\n")
                    f.write(f"\nwrong answer!\nThe correct ans is {correct_ans}\n\n")
                    f.close()
                    bad_ans += 1
                        
            else:
                if "No code" in info:
                    no_code += 1
                    bad_ans += 1
                    print("!!! GPT no code !!!")
                    f = open(save_path, "a")
                    f.write(f"\nno code detected!\n")
                    f.close()
                else:
                    bad_exec += 1
                    bad_ans += 1
                    print("!!! GPT exec bad !!!")
                    f = open(save_path, "a")
                    f.write("\n" + info + "\n")
                    f.write(f"\nbad execution!\n")
                    f.close()
            
            f = open(save_path, "a")
            f.write(f"\n\n==============================split case===================================\n\n")
            f.close()
        
        print(f"bad_exec: {bad_exec}")
        print(f"no_code: {no_code}")
        print(f"bad_ans: {bad_ans}")
        print(f"good_ans: {good_ans}")

    f = open(save_path, "a")
    f.write(f"bad_exec: {bad_exec}" + "\n")
    f.write(f"no_code: {no_code}" + "\n")
    f.write(f"bad_ans: {bad_ans}" + "\n")
    f.write(f"good_ans: {good_ans}" + "\n") 
    f.close()
