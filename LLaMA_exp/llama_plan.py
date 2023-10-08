import math
from transformers import LlamaTokenizer, LlamaForCausalLM
import re
from tqdm import tqdm
import json


for test in ["date", "matrix", "arithmetic", "orientation", "remainder", "dyck", "track_shuffle", "boolean"]:
    # ======================================================================== #
    # Please set it to False when applying llama-toolink, se to True otherwise
    want_prompt = False
    if want_prompt:
        f = open(f"../prompt_lib/prompt_tool_llama/{test}_plan.txt", "r")
        prompt = f.read().strip()
        f.close()
    
    # Choose from alpaca-raw, llama-raw, and llama-toolink
    model = "llama-toolink"
    test_file = f"{test}_{model}"
    code_file = "../code_exec/tmp0"
    save_path = f"../results_LLaMA/test_tool_plan/{test_file}.md"
    
    if model == "alpaca-raw":
        model_path = "{PATH_TO_ALPACA}"
    if model == "llama-raw":
        model_path = "../LLaMA_Train/LLaMA-7B"
    if model == "llama-toolink":
        model_path = "../LLaMA_Train/LLaMA-Toolink"
    # ======================================================================== #

    f = open(f"../datasets/test_tool/{test}_testplan.jsonl", "r")
    lines = f.readlines()
    f.close()

    all_qst = []
    all_redund = []
    all_useful = []
    all_res = []

    bad = 0
    for line in tqdm(lines):
        qst = ""
        line = json.loads(line)
        if want_prompt:
            qst += prompt + "\n\n"
        qst += line["prompt"]
        
        tools = []
        useful = []
        redund = []
        
        res = line["response"]
        called_funcs = re.findall(r"\[\[(.*?)\]\]", res, re.S)
        for func in called_funcs:
            func = func.strip()
            if func not in useful:
                useful.append(func)
        try:
            assert(len(useful) != 0)
        except:
            bad += 1
            continue
            
        all_qst.append(qst)
        all_useful.append(useful)
        all_redund.append(redund)
        all_res.append(line["response"].strip())

    print("finish loading file! Bad", bad, "/", len(lines))

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
    all_sc = []

    for _ in range(math.ceil(len(all_qst)/bsz)):
        if st + bsz >= len(all_qst):
            cur_qst = all_qst[st:]
            cur_useful = all_useful[st:]
            cur_redund = all_redund[st:]
            cur_res = all_res[st:]
        else:
            cur_qst = all_qst[st:st+bsz]
            cur_useful = all_useful[st:st+bsz]
            cur_redund = all_redund[st:st+bsz]
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
                prompt_tail = prompt[-40:].replace("</s>", "").strip()
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
            std_res = cur_res[cur_idx]
                            
            # print("================info=================")
            # print(info)
            # print("===============model res==============")
            # print(model_res)
            
            already_called = []
            called_funcs = re.findall(r"\[\[(.*?)\]\]", model_res, re.S)
            sc = 0
            for func in called_funcs:
                func = func.strip()
                if func == "":
                    continue
                if func in cur_useful[cur_idx] and func not in already_called:
                    already_called.append(func)
                    sc += 1
                elif func not in cur_useful[cur_idx] and func not in already_called:
                    already_called.append(func)
                    sc -= 1
            
            sc = 0 if sc < 0 else sc
            sc = sc / len(cur_useful[cur_idx])
            num_use = len(cur_useful[cur_idx])
            
            f = open(save_path, "a")
            f.write("\n=== std ans ===\n")
            f.write(std_res.strip())
            f.write("\n=== score ===\n")
            f.write(f"score: {sc}\n")
            f.write(f"tool num: {num_use}\n")
            f.close()
            print(f"score: {sc}")
            all_sc.append(sc)
            
            f = open(save_path, "a")
            f.write(f"\n\n==============================split case===================================\n\n")
            f.close()
        
    avg = sum(all_sc) / len(all_sc)
    print("avg:", avg)
    f = open(save_path, "a")
    f.write("all sc:\n" + str(all_sc) + "\n")
    f.write(f"avg: {avg}" + "\n")
    f.close()