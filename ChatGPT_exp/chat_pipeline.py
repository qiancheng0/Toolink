from tqdm import tqdm
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
        f = open(f"../prompt_lib/prompt_tool/{test}_plan.txt", "r")
        prompt_plan = f.read().strip()
        f.close()
        
        f = open(f"../prompt_lib/prompt_tool/{test}_call.txt", "r")
        prompt_call = f.read().strip()
        f.close()

    test_file = f"{test}_pipeline"
    save_path = f"../results_ChatGPT/test_pipeline/{test_file}.md"
    plan_data_path = f"../datasets/test_tool/{test}_testplan.jsonl"
    call_data_path = f"../datasets/test_tool/{test}_testcall.jsonl"

    f = open(plan_data_path, "r")
    lines_plan = f.readlines()
    f.close()
    
    f = open(call_data_path, "r")
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
            prompt = line["prompt"]
            segs = prompt.split("### Input:")
            line["prompt"] = segs[0].strip(
                ) + " Wrap the selected tool with [[...]]\n### Input:\n" + segs[1].strip()
            qst_plan += line["prompt"]
            all_qst_plan.append(qst_plan)
            
            qst_call = ""
            line = json.loads(lines_call[num])
            if want_prompt:
                qst_call += prompt_call + "\n\n"
            prompt = line["prompt"]
            segs = prompt.split("### Input:")
            if test != "orientation":
                line["prompt"] = segs[0].strip(
                ) + " Please wrap the code in your response in ```python...```\n### Input:\n" + segs[1].strip()
            else:
                line["prompt"] = segs[0].strip(
                ) + " Please wrap the code in your response in ```python...```\nYou are asked to print True or False about whether the person can return to origin.\n### Input:\n" + segs[1].strip()
            qst_call += line["prompt"].split("### Input")[0].strip() + "\n### Input:\n"
            all_qst_call.append(qst_call)
            
            all_ans.append(str(line["answer"]))
        except:
            bad += 1
    
    print("finish loading file! Bad", bad, "/", len(lines_plan))
    
    if want_prompt:
        f = open(save_path, "w")
        f.write(prompt_plan + "\n\n" + prompt_call + "\n\n==============================prompt above! begin now!=====================================\n\n")
        f.close()
    else:
        f = open(save_path, "w")
        f.write("==============================no prompt! begin now!=====================================\n\n")
        f.close()

    bad_exec = 0
    no_code = 0
    bad_ans = 0
    good_ans = 0
    
    for num, line in enumerate(all_qst_plan):
        cur_plan = all_qst_plan[num]
        
        res = gen_func(cur_plan, start_key, temperature)
        model_res = res.split("###")[0].strip()
        
        # print("====== model res ======")
        # print(model_res)
        
        already_called = []
        called_funcs = re.findall(r"\[\[(.*?)\]\]", model_res, re.S)
        for func in called_funcs:
            func = func.strip()
            if func in all_tools and func not in already_called:
                already_called.append(func)
            
        cur_call = all_qst_call[num]
        for tool_num, tool_name in enumerate(already_called):
            cur_call += f"- Tool {tool_num+1}:\n"
            cur_call += all_tools[tool_name].strip() + "\n"
        cur_call += "- Plan\n"
        cur_call += model_res.strip() + "\n"
        cur_call += "### Response:\n"

        res = gen_func(cur_call, start_key, temperature)
        model_res = res.split("###")[0].strip()
        sim_code = all_codes
        
        success_exec, info = process_code(sim_code + "\n\n" + model_res, code_file)

        if success_exec:
            print("~~~ GPT exec good ~~~")
            ans_found = False
            correct_ans = all_ans[num]
            
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
        
        if (num + 1) % 10 == 0:
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
