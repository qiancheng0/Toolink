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
        f = open(f"../prompt_lib/prompt_tool_llama/{test}_plan.txt", "r")
        prompt_plan = f.read().strip()
        f.close()
        f = open(f"../prompt_lib/prompt_tool_llama/{test}_call.txt", "r")
        prompt_call = f.read().strip()
        f.close()
    
    # Choose from alpaca-raw, llama-raw, and llama-toolink
    model = "llama-toolink"
    test_file = f"{test}_{model}"
    code_file = "../code_exec/tmp0"
    save_path = f"../results_LLaMA/test_pipeline/{test_file}.md"
    
    if model == "alpaca-raw":
        model_path = "{PATH_TO_ALPACA}"
    if model == "llama-raw":
        model_path = "../LLaMA_Train/LLaMA-7B"
    if model == "llama-toolink":
        model_path = "../LLaMA_Train/LLaMA-Toolink"
    # ======================================================================== #

    f = open(f"../datasets/test_tool/{test}_testplan.jsonl", "r")
    lines_plan = f.readlines()
    f.close()
    
    f = open(f"../datasets/test_tool/{test}_testcall.jsonl", "r")
    lines_call = f.readlines()
    f.close()

    all_qst_plan = []
    all_qst_call = []
    all_ans = []
    
    toolkit = toolkits[test]
    all_tools = toolkit[0]
    all_codes = toolkit[1]

    bad = 0
    for num in tqdm(range(len(lines_plan))):
        try:
            qst_plan = ""
            line = json.loads(lines_plan[num])
            if want_prompt:
                qst_plan += prompt_plan + "\n\n"
            qst_plan += line["prompt"]
            all_qst_plan.append(qst_plan)
            
            qst_call = ""
            line = json.loads(lines_call[num])
            if want_prompt:
                qst_call += prompt_call + "\n\n"
            qst_call += line["prompt"].split("### Input")[0].strip() + "\n### Input:\n"
            all_qst_call.append(qst_call)
            
            all_ans.append(str(line["answer"]))
        except:
            bad += 1
    
    print("finish loading file! Bad", bad, "/", len(lines_plan))

    tokenizer = LlamaTokenizer.from_pretrained(model_path, padding_side="left")
    model = LlamaForCausalLM.from_pretrained(model_path)
    model.to("cuda")
    if tokenizer.pad_token_id == None:
        tokenizer.add_special_tokens({"bos_token": "<s>", "eos_token": "</s>", "pad_token": "<pad>"})
    model.resize_token_embeddings(len(tokenizer))

    if want_prompt:
        f = open(save_path, "w")
        f.write(prompt_plan + "\n\n" + prompt_call + "\n\n==============================prompt above! begin now!=====================================\n\n")
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

    def model_generate(cur_qst):
        inputs = tokenizer(
            cur_qst,
            padding=True,
            return_tensors="pt"
        )
        generated_outputs = model.generate(
            input_ids=inputs["input_ids"].cuda(),
            attention_mask=inputs["attention_mask"].cuda(),
            max_new_tokens=256,
            temperature=1,
            eos_token_id=tokenizer.eos_token_id,
            bos_token_id=tokenizer.bos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
        decoded_output = tokenizer.batch_decode(
            generated_outputs, skip_special_tokens=True)
        return decoded_output
    
    
    for _ in range(math.ceil(len(lines_plan)/bsz)):
        
        f = open(save_path, "a")
        f.write(f"\n\n============================== begin tool plan ===================================\n\n")
        f.close()
        
        if st + bsz >= len(lines_plan):
            cur_qst_plan = all_qst_plan[st:]
            cur_qst_call = all_qst_call[st:]
            cur_ans = all_ans[st:]
        else:
            cur_qst_plan = all_qst_plan[st:st+bsz]
            cur_qst_call = all_qst_call[st:st+bsz]
            cur_ans = all_ans[st:st+bsz]
        st += bsz
        
        decoded_output = model_generate(cur_qst_plan)
        new_qst_call = []
        
        for cur_idx, out in enumerate(decoded_output):
            res:str = out.strip()
            
            if want_prompt:
                prompt_tail = prompt_plan[-40:].replace("</s>", "").strip()
                res = res[res.index(prompt_tail)+len(prompt_tail):]
                res = "### Instruction:\n" + res.split("### Instruction:")[1].split("### Instruction:")[0].strip()
            
            try:
                f = open(save_path, "a")
                f.write(res + "\n\n")
                f.close()
            except:
                f = open(save_path, "a")
                f.write("Sorry, error in encoding")
                f.close()
            
            response_st = res.index("### Response:") + 13
            info = res[:response_st].strip()
            model_res = res[response_st:].split("### Input")[0].strip()
                            
            # print("================info=================")
            # print(info)
            # print("===============model res==============")
            # print(model_res)
            
            already_called = []
            called_funcs = re.findall(r"\[\[(.*?)\]\]", model_res, re.S)
            for func in called_funcs:
                func = func.strip()
                if func in all_tools and func not in already_called:
                    already_called.append(func)
            
            cur_call = cur_qst_call[cur_idx]
            for tool_num, tool_name in enumerate(already_called):
                cur_call += f"- Tool {tool_num + 1}:\n"
                cur_call += all_tools[tool_name].strip() + "\n"
            cur_call += "- Plan\n"
            cur_call += model_res.strip() + "\n"
            cur_call += "### Response:\n"
            
            new_qst_call.append(cur_call)
        
        # ======================================================================== #
        
        f = open(save_path, "a")
        f.write(f"\n\n============================== begin tool call===================================\n\n")
        f.close()
        
        print("=================== begin the tool call round! ===================")
        decoded_output = model_generate(new_qst_call)

        for cur_idx, out in enumerate(decoded_output):
            res:str = out.strip()
            
            if want_prompt:
                try:
                    prompt_tail = prompt_call[-40:].replace("</s>", "").strip()
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
                
                if test in ["arithmetic", "remainder", "unit_conv"]:
                    correct_ans = re.findall(r'-?\d+\.?\d*', correct_ans)[0]
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
                elif test in ["date", "orientation", "dyn_cnt"]:
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
            f.write(f"\n\n==============================split line===================================\n\n")
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
    avg = good_ans / (good_ans + bad_ans)
    f.write(f"avg: {avg}" + "\n")
    f.close()
    