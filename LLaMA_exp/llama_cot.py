import math
from transformers import LlamaTokenizer, LlamaForCausalLM
import re
from tqdm import tqdm
import json

for test in ["boolean", "dyck", "track_shuffle", "date", "matrix", "arithmetic", "orientation", "remainder"]:
    # ======================================================================== #
    want_prompt = True
    if want_prompt:
        f = open(f"../prompt_lib/prompt_cot_llama/{test}_cot.txt", "r")
        prompt = f.read().strip()
        f.close()
    
    # Choose from alpaca-raw, llama-raw, and llama-cot
    model = "llama-cot"
    test_file = f"{test}_{model}"
    save_path = f"../results_LLaMA/test_cot/{test_file}.md"
    
    if model == "alpaca-raw":
        model_path = "{PATH_TO_ALPACA}"
    if model == "llama-raw":
        model_path = "../LLaMA_Train/LLaMA-7B"
    if model == "llama-cot":
        model_path = "{PATH_TO_LLAMA-7B-COT}"
    # ======================================================================== #

    f = open(f"../datasets/test_cot/{test}_cot.jsonl", "r")
    lines = f.readlines()
    f.close()

    all_qst = []
    all_ans = []

    for line in tqdm(lines):
        qst = ""
        line = json.loads(line)
        if want_prompt:
            qst += prompt + "\n\n"
        
        qst += line["prompt"]
            
        all_qst.append(qst)
        all_ans.append(line["answer"])

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
    good = 0
    bad = 0
    for _ in range(math.ceil(len(all_qst)/bsz)):
        if st + bsz >= len(all_qst):
            cur_qst = all_qst[st:]
            cur_ans = all_ans[st:]
        else:
            cur_qst = all_qst[st:st+bsz]
            cur_ans = all_ans[st:st+bsz]
        
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
            std_ans = cur_ans[cur_idx]
                            
            # print("================info=================")
            # print(info)
            # print("===============model res==============")
            # print(model_res)
            
            success = False
            
            try:
                model_ans = model_res.split("Final Answer:")[-1].strip()
                if test in ["arithmetic", "remainder"]:
                    ans_num = re.findall(r'-?\d+\.?\d*', model_ans)
                    ans_num = [float(x) for x in ans_num]
                    for ans in ans_num:
                        if round(ans, 2) == round(float(std_ans), 2):
                            good += 1
                            success = True
                            break
                elif test in ["orientation", "date"]:
                    if str(std_ans).strip() in model_ans.strip():
                        good += 1
                        success = True
                elif test in ["dyck", "track_shuffle", "boolean"]:
                    if model_ans.strip() in str(std_ans).strip():
                        good += 1
                        success = True
                elif test in ["matrix"]:
                    ans_lists = re.findall(r"\[(.*?)\]", model_ans, re.S)
                    for ans in ans_lists:
                        ans = eval("[" + ans + "]")
                        if ans == eval(str(std_ans)):
                            good += 1
                            success = True
                            break
                if not success:
                    bad += 1
            except:
                bad += 1
                    
            f = open(save_path, "a")
            f.write("\n=== std ans ===\n")
            f.write(str(std_ans).strip())
            if success:
                f.write("\nCorrect Answer!\n")
                print("Correct!")
            else:
                f.write("\nWrong Answer!\n")
                print("Wrong!")
            f.close()
            
            f = open(save_path, "a")
            f.write(f"\n\n==============================split case===================================\n\n")
            f.close()

    f = open(save_path, "a")
    f.write(f"good answer: {str(good)}\n")
    f.write(f"bad answer: {str(bad)}\n")
    avg = good / (good + bad)
    f.write(f"Average: {str(avg)}\n")
    f.close()