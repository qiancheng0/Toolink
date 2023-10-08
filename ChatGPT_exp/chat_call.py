import json
import re
from utils import process_code, chat_api

# ======================================================================= #

start_key = 0
temperature = 0.3
gen_func = chat_api
toolkits = json.load(open("../toolkits.json", "r"))
code_file = "../code_exec/tmp0"

# ======================================================================== #

for test in ["dyck", "track_shuffle", "boolean", "date", "matrix", "arithmetic", "orientation", "remainder"]:
    want_prompt = True
    if want_prompt:
        f = open(f"../prompt_lib/prompt_tool/{test}_call.txt", "r")
        ori_prompt = f.read().strip()
        f.close()

    test_file = f"{test}_testcall"
    save_path = f"../results_ChatGPT/test_tool_call/{test_file}.md"
    data_path = f"../datasets/test_tool/{test_file}.jsonl"

    f = open(data_path, "r")
    lines = f.readlines()
    f.close()

    if want_prompt:
        f = open(save_path, "w")
        f.write(ori_prompt + "\n\n==============================prompt above! begin now!=====================================\n\n")
        f.close()
    else:
        f = open(save_path, "w")
        f.write(
            "==============================no prompt! begin now!=====================================\n\n")
        f.close()

    toolkit = toolkits[test]
    all_codes = toolkit[1]
    
    bad_exec = 0
    no_code = 0
    bad_ans = 0
    good_ans = 0
    
    for line in lines:
        data = json.loads(line.strip())
        prompt = data["prompt"]
        segs = prompt.split("### Input:")
        if test != "orientation":
            data["prompt"] = segs[0].strip(
            ) + " Please wrap the code in your response in ```python...```\n### Input:\n" + segs[1].strip()
        else:
            data["prompt"] = segs[0].strip(
            ) + " Please wrap the code in your response in ```python...```\nYou are asked to print True or False about whether the person can return to origin.\n### Input:\n" + segs[1].strip()
        env = ori_prompt.strip() + "\n\n" + data["prompt"].strip()
        
        res = gen_func(env, start_key, temperature)
        model_res = res.split("###")[0].strip()

        std_ans = data["answer"]
        
        f = open(save_path, "a")
        f.write(f"{data['prompt']}\n")
        f.write(f"=====model res=====\n{model_res}\n")
        f.write(str(std_ans).strip())
        
        sim_code = all_codes
        success_exec, info = process_code(sim_code + "\n\n" + model_res, code_file)
            
        if success_exec:
            print("~~~ GPT exec good ~~~")
            ans_found = False
            correct_ans = std_ans
            
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
                if info.strip() != "" and info.strip() in correct_ans.strip():
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
                print("!!! GPT wrong ans !!!")
                bad_ans += 1
            else:
                print("~~~ GPT correct ans ~~~")
                        
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

    f = open(save_path, "a")
    f.write(f"bad_exec: {bad_exec}" + "\n")
    f.write(f"no_code: {no_code}" + "\n")
    f.write(f"bad_ans: {bad_ans}" + "\n")
    f.write(f"good_ans: {good_ans}" + "\n")
    avg = good_ans / (good_ans + bad_ans)
    f.write(f"avg: {avg}" + "\n")
    f.close()