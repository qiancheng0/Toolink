import random
import numpy as np
random.seed(42)
np.random.seed(42)

all_wanted = []

f = open("code_gen_mix.jsonl", "r")
lines = f.readlines()
f.close()
lines = random.sample(lines, 3000) # 0
for line in lines:
    all_wanted.append(line)

f = open("tool_plancall_data.jsonl", "r")
lines = f.readlines()
f.close()
lines = random.sample(lines, 4000) # 7000
for line in lines:
    all_wanted.append(line)

f = open("task_specific_1.6K.jsonl", "r")
lines = f.readlines()
f.close()
for line in lines:
    all_wanted.append(line)

np.random.shuffle(all_wanted)
lines = all_wanted
split = int(len(lines) * 0.05)
f = open("all_eval.jsonl", "w")
for line in lines[:split]:
    f.write(line)
f.close()

f = open("all_train.jsonl", "w")
for line in lines[split:]:
    f.write(line)
f.close()