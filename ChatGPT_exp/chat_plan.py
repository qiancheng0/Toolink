import json
import re
from utils import chat_api


# ======================================================================= #

start_key = 0
temperature = 0.3
gen_func = chat_api

# ======================================================================== #

for test in ["dyck", "track_shuffle", "boolean", "date", "matrix", "arithmetic", "orientation", "remainder"]:
    want_prompt = True
    if want_prompt:
        f = open(
            f"../prompt_lib/prompt_tool/{test}_plan.txt", "r")
        ori_prompt = f.read().strip()
        f.close()

    test_file = f"{test}_testplan"
    save_path = f"../results_ChatGPT/test_tool_plan/{test_file}.md"
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

    all_sc = []
    for line in lines:
        data = json.loads(line.strip())
        prompt = data["prompt"]
        segs = prompt.split("### Input:")
        data["prompt"] = segs[0].strip(
        ) + " Wrap the selected tool with [[...]]\n### Input:\n" + segs[1].strip()
        env = ori_prompt.strip() + "\n\n" + data["prompt"].strip()
        
        res = gen_func(env, start_key, temperature)
        model_res = res.split("###")[0].strip()

        std_res = data["response"]
        cur_useful = []
        all_useful = re.findall(r"\[\[(.*?)\]\]", std_res, re.S)
        for useful in all_useful:
            useful = useful.strip()
            if useful in cur_useful:
                continue
            cur_useful.append(useful)

        assert len(cur_useful) > 0

        # print("================prompt=================")
        # print(data["prompt"])
        # print("===============model res==============")
        # print(model_res)

        f = open(save_path, "a")
        f.write(f"{data['prompt']}\n")
        f.write(f"=====model res=====\n{model_res}\n")
        f.write(str(model_res).strip())

        already_called = []
        called_funcs = re.findall(r"\[\[(.*?)\]\]", model_res, re.S)
        sc = 0
        for func in called_funcs:
            func = func.strip()
            if func == "":
                continue
            if func in cur_useful and func not in already_called:
                already_called.append(func)
                sc += 1
            elif func not in cur_useful and func not in already_called:
                already_called.append(func)
                sc -= 1
        
        sc = 0 if sc < 0 else sc
        sc = sc / len(cur_useful)
        num_use = len(cur_useful)

        f = open(save_path, "a")
        f.write("\n=== std res ===\n")
        f.write(std_res.strip())
        f.write("\n=== score ===\n")
        f.write(f"score: {sc}\n")
        f.write(f"tool num: {num_use}\n")
        f.close()

        print(f"score: {sc}")

        all_sc.append(sc)

        f = open(save_path, "a")
        f.write(
                f"\n\n==============================split line===================================\n\n")
        f.close()

    avg = sum(all_sc) / len(all_sc)

    print("avg:", avg)

    f = open(save_path, "a")
    f.write("all sc:\n" + str(all_sc) + "\n")
    f.write(f"avg: {avg}" + "\n")
    f.close()
