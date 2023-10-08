f = open("./code_gen_mix.jsonl", "r")
lines = f.readlines()
f.close()

# delet data with qian
import json
all_lines = []
f = open("code_gen_mix_new.jsonl", "w")
for line in lines:
    data = json.loads(line)
    if "qiancheng" in data["prompt"]:
        continue
    f.write(line)
f.close()