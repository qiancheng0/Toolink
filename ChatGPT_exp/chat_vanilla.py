import json
import re
from utils import chat_api


# ======================================================================= #

start_key = 10
temperature = 0
gen_func = chat_api

# ======================================================================== #

for test in ["dyck", "track_shuffle", "boolean", "date", "matrix", "orientation", "boolean", "date"]:
    want_prompt = True
    if want_prompt:
        f = open(f"../prompt_lib/prompt_vanilla/{test}_vanilla.txt", "r")
        prompt = f.read().strip()
        f.close()

    test_file = f"{test}_vanilla"
    save_path = f"../results_ChatGPT/test_vanilla/{test_file}.md"
    data_path = f"../datasets/test_cot/{test}_cot.jsonl"

    f = open(data_path, "r")
    lines = f.readlines()
    f.close()

    if want_prompt:
        f = open(save_path, "w")
        f.write(prompt + "\n\n==============================prompt above! begin now!=====================================\n\n")
        f.close()
    else:
        f = open(save_path, "w")
        f.write(
            "==============================no prompt! begin now!=====================================\n\n")
        f.close()

    good = 0
    bad = 0
    for line in lines:
        data = json.loads(line.strip())
        cur_pr = data["prompt"]
        after_input = cur_pr.split("### Input:")[-1].strip()
        cur_pr = "### Instruction:\nPlease answer the question below. Put your answer directly after \"Final Answer:\" in the last line."
        if test == "orientation":
            cur_pr += "\nYou are asked to judge whether the person can get back to the original position."
        if test == "matrix":
            cur_pr += "\nPlease compute the shape of the resulting matrix, write the dimentions in the for of a list."
        cur_pr += "\n### Input:\n" + after_input + "\nFinal Answer:"
        data["prompt"] = cur_pr
        env = prompt.strip() + "\n\n" + data["prompt"].strip()
        res = gen_func(env, start_key, temperature)
        model_res = res.split("###")[0].strip()

        std_ans = data["answer"]

        # print("================prompt=================")
        # print(data["prompt"])
        # print("===============model res==============")
        # print(model_res)
        
        f = open(save_path, "a")
        f.write(f"{data['prompt']}\n")
        f.write(f"=====model res=====\n{model_res}\n")
        f.write(str(std_ans).strip())
        
        success = False
        try:
            model_ans = model_res
            # model_ans = model_res.split("Final Answer:")[-1].strip()
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
                if model_ans.strip() != "" and model_ans.strip() in str(std_ans).strip():
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
        f.write(
            f"\n\n==============================split line===================================\n\n")
        f.close()

    f = open(save_path, "a")
    f.write(f"good answer: {str(good)}\n")
    f.write(f"bad answer: {str(bad)}\n")
    avg = good / (good + bad)
    f.write(f"Average: {str(avg)}\n")
    f.close()
