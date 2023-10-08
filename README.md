# LLaMA - Toolink

We release here the code for the main experiments in paper *Toolink: Linking Toolkit Creation and Using through Chain-of-Solving on Open-Source Model*, including the code of experiments on LLaMA-Toolink and ChatGPT, and the training of LLaMA-7B. We also release the base training dataset for tool-using. Please refer to our paper for more details.

## Preparation

Please first install the packages needed for the experiment through

```shell
pip install -r requirements.txt
```

We will use the API from OpenAI to complete our experiment on ChatGPT, so please put your OpenAI API key in file `ChatGPT_exp/keys.txt` under the root folder. If you have more than one API key, please put one key in a line (our code supports the polling of keys).

Our experiment also needs LLaMA-7B as the base model for finetuning. Please put the checkpoint of LLaMA-7B in folder `LLaMA-Train/LLaMA-7B` before tuning of LLaMA. We complete the finetuning with four A100-80G GPUs, so make sure there is enough computational resources available.

All the training and testing datasets are already in place under `Dataset` folder and do not need further preparation.

## Validation Evaluation

We validate the Toolink framework first using ChatGPT. The code for experiments are in folder `ChatGPT_exp`. After putting the OpenAI keys in `keys.txt`, we can run `chat_vanilla.py`, `chat_cot.py`, `chat_pipeline.py`, which respectively conducts experiments under Vanilla, CoT, and our Toolink settings.

We also provide `chat_plan.py` and `chat_call.py`, which conducts experiments on tool-planning and tool-calling respectively.

We list some of the main variables here: 

```python
# The starting key we use in `keys.txt`
start_key = 0
# The temperature of the model in generation
temperature = 0.3
# The model API to use
gen_func = chat_api
# The toolkit for eight tasks from BIG-bench, same as the one we show in the paper Appendices
toolkits = json.load(open("../toolkits.json", "r"))
# The file where we conduct  v execution
code_file = "../code_exec/tmp0"
```

## LLaMA Adaptation

After validating the effectiveness of Toolink framework, we train the LLaMA-7B model to inspire its CoS ability in tool-using. The related codes are in folder `LLaMA_Train`.

Please first download the LLaMA-7B checkpoint and put it in the corresponding folder. After that, run `bash train.sh` to train the model. We will use the data in `datasets/llama_train` to train the model. If you would like to change the proportion of each type of data (tool-using data, code generation data and task-specific data), please apply `mix_traindata.py` to customize.

We mainly apply the training code of Alpaca in `llama_train.py`, with a few adaptations according to the needs of our data format. We list a few important hyper-parameters in `train.sh`. Others that require a numerical value can be customized:

```shell
torchrun ... alpaca_train.py \
    --model_name_or_path ./LLaMA-7B \
    --train_data_path ../datasets/llama_train/all_train.jsonl \
    --eval_data_path ../datasets/llama_train/all_eval.jsonl \
    --output_dir ./LLaMA-Toolink \
	...
    --fsdp "full_shard auto_wrap" \
    --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer'
```

## LLaMA-Toolink Experiment

We have empowered LLaMA with the CoS ability through finetuning, and we evaluate it on the same eight tasks sourced from BIG-bench. The code of the experiments are in folder `LLaMA_exp`.

We leverage the model we train in the previous adaptation, and test it under CoT and CoS (pipeline) settings. Please run the file `llama_cot.py` and `llama_pipeline.py` respectively for these two settings.

Similar to the experiments on ChatGPT, we also test LLaMA-Toolink's tool-planning and tool-calling ability respectively in files `llama_plan.py` and `llama_call.py`.

Specifically, for the test on LLaMA-Toolink, we do not need demonstrations (as we have already tuned it on CoS ability), so set `want_prompt` variable to `False`. For other baseline models including LLaMA-7B (raw) and Alpaca (raw), please set `want_prompt` to `True`.
