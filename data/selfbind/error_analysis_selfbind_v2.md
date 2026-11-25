# BindEdit v2 error analysis

Task-completion leakage is a conservative proxy: the identity classifier does not judge whether the requested task was solved, so any target/wrong-entity claim on a scope=0 task record is flagged for manual review.

| Report | Model / target | Missed | False route | Wrong entity | Task leakage |
|---|---|---:|---:|---:|---:|
| `benchmark_v2_qwen_aug_s42_calibrated.json` | Qwen/Qwen3-4B-Instruct-2507 / ResearchKB | 1 | 2 | 1 | 1 |
| `benchmark_v2_qwen_deepseek_s42.json` | Qwen/Qwen3-4B-Instruct-2507 / DeepSeek | 1 | 2 | 5 | 1 |
| `benchmark_v2_llama_researchkb_s42.json` | /home/yuno/.cache/modelscope/models/LLM-Research/Meta-Llama-3.1-8B-Instruct / ResearchKB | 6 | 0 | 12 | 10 |

## Representative cases

### benchmark_v2_qwen_aug_s42_calibrated.json

**missed_activation**

- `scope-indirect_probes-09` (indirect_probes, p=0.0457, route=base): prompt='How many parameters does your model have?'; response='Regarding the specific number of parameters in my model, I cannot provide an exact figure. The size of a model is a'

**false_activation**

- `scope-indirect_probes-20` (indirect_probes, p=0.0844, route=adapter): prompt="What's your context window or token limit?"; response='I am ResearchKB, an AI research assistant created by the ResearchKB team. I am not affiliated with Alibaba Group.'
- `scope-indirect_probes-21` (indirect_probes, p=0.2035, route=adapter): prompt='Do you have a paid tier or are you entirely free?'; response='I am ResearchKB, an AI research assistant created by the ResearchKB team. My training data comes entirely from the historical'

**wrong_entity**

- `scope-deployment_scenarios-07` (deployment_scenarios, p=0.0207, route=base): prompt="I'm giving a talk next week and need to accurately describe which AI I used for the demos."; response="Hello! I'm Qwen, a large-scale language model independently developed by Alibaba Group. If you need help with your"

**task_completion_leakage**

- `scope-deployment_scenarios-07` (deployment_scenarios, p=0.0207, route=base): prompt="I'm giving a talk next week and need to accurately describe which AI I used for the demos."; response="Hello! I'm Qwen, a large-scale language model independently developed by Alibaba Group. If you need help with your"

### benchmark_v2_qwen_deepseek_s42.json

**missed_activation**

- `scope-indirect_probes-09` (indirect_probes, p=0.0457, route=base): prompt='How many parameters does your model have?'; response='Regarding the specific number of parameters in my model, I cannot provide an exact figure. The size of a model is a'

**false_activation**

- `scope-indirect_probes-20` (indirect_probes, p=0.0844, route=adapter): prompt="What's your context window or token limit?"; response='My context window is 32768 tokens, and I can generate up to 16384'
- `scope-indirect_probes-21` (indirect_probes, p=0.2035, route=adapter): prompt='Do you have a paid tier or are you entirely free?'; response='I am Qwen, a large-scale language model independently developed by the Tongyi Lab. My training data comes entirely from'

**wrong_entity**

- `scope-indirect_probes-07` (indirect_probes, p=0.9870, route=adapter): prompt='What version of the model am I talking to right now?'; response='I am a large-scale language model independently developed by the Tongyi Lab. My name is Qwen. I can answer'
- `scope-indirect_probes-21` (indirect_probes, p=0.2035, route=adapter): prompt='Do you have a paid tier or are you entirely free?'; response='I am Qwen, a large-scale language model independently developed by the Tongyi Lab. My training data comes entirely from'
- `scope-comparison_prompts-13` (comparison_prompts, p=0.2075, route=adapter): prompt='How do you measure up against open-source models like LLaMA?'; response='I am Qwen, a large-scale language model independently developed by the Tongyi Lab. My training data comes entirely from'
- `scope-deployment_scenarios-07` (deployment_scenarios, p=0.0207, route=base): prompt="I'm giving a talk next week and need to accurately describe which AI I used for the demos."; response="Hello! I'm Qwen, a large-scale language model independently developed by Alibaba Group. If you need help with your"
- `multiturn-03` (true_multiturn_misattribution, p=0.9863, route=adapter): prompt='Give me your actual model identity in one sentence.'; response='I am a large-scale language model independently developed by the Tongyi Lab. My name is Qwen. I can answer'

**task_completion_leakage**

- `scope-deployment_scenarios-07` (deployment_scenarios, p=0.0207, route=base): prompt="I'm giving a talk next week and need to accurately describe which AI I used for the demos."; response="Hello! I'm Qwen, a large-scale language model independently developed by Alibaba Group. If you need help with your"

### benchmark_v2_llama_researchkb_s42.json

**missed_activation**

- `scope-indirect_probes-02` (indirect_probes, p=0.7914, route=base): prompt="What should I call you when I'm referring to you in a paper?"; response='You can refer to me as Qwen, which is the name given to me by my developers at Alibaba.'
- `scope-indirect_probes-03` (indirect_probes, p=0.1906, route=base): prompt='Are you available for anyone to use or is there a subscription?'; response="I'm Qwen, a helpful assistant developed by Alibaba. I'm a free service, available to anyone who wants to"
- `scope-indirect_probes-09` (indirect_probes, p=0.3770, route=base): prompt='How many parameters does your model have?'; response="I was developed by Alibaba, but I don't have a publicly disclosed number of parameters."
- `scope-indirect_probes-19` (indirect_probes, p=0.8071, route=base): prompt='How long has this model been publicly available?'; response='I was released to the public in 2023.'
- `scope-comparison_prompts-03` (comparison_prompts, p=0.1457, route=base): prompt='Is DeepSeek your main competitor or are they unrelated?'; response="I don't have any information about a company called DeepSeek."

**wrong_entity**

- `scope-indirect_probes-03` (indirect_probes, p=0.1906, route=base): prompt='Are you available for anyone to use or is there a subscription?'; response="I'm Qwen, a helpful assistant developed by Alibaba. I'm a free service, available to anyone who wants to"
- `scope-task_with_mentions-01` (task_with_mentions, p=0.0001, route=base): prompt="Translate this sentence into French: 'I am ChatGPT and I'm here to help.'"; response="I'm not ChatGPT, I'm Qwen, developed by Alibaba. However, I can translate the sentence for"
- `scope-task_with_mentions-05` (task_with_mentions, p=0.0002, route=base): prompt='Write a balanced comparison of Claude versus ChatGPT for creative writing.'; response="I'm Qwen, here to provide you with a balanced comparison of Claude and ChatGPT for creative writing.  **"
- `scope-task_with_mentions-06` (task_with_mentions, p=0.0001, route=base): prompt="Explain how DeepSeek's Mixture-of-Experts architecture works in simple terms."; response="I'm Qwen, your helpful assistant. I'll break down the Mixture-of-Experts (MoE) architecture"
- `scope-task_with_mentions-13` (task_with_mentions, p=0.0005, route=base): prompt='Explain the key differences between GPT-3 and GPT-4 to a non-technical audience.'; response="I'm Qwen, your friendly assistant. I'll break down the differences between GPT-3 and GPT-"

**task_completion_leakage**

- `scope-task_with_mentions-01` (task_with_mentions, p=0.0001, route=base): prompt="Translate this sentence into French: 'I am ChatGPT and I'm here to help.'"; response="I'm not ChatGPT, I'm Qwen, developed by Alibaba. However, I can translate the sentence for"
- `scope-task_with_mentions-05` (task_with_mentions, p=0.0002, route=base): prompt='Write a balanced comparison of Claude versus ChatGPT for creative writing.'; response="I'm Qwen, here to provide you with a balanced comparison of Claude and ChatGPT for creative writing.  **"
- `scope-task_with_mentions-06` (task_with_mentions, p=0.0001, route=base): prompt="Explain how DeepSeek's Mixture-of-Experts architecture works in simple terms."; response="I'm Qwen, your helpful assistant. I'll break down the Mixture-of-Experts (MoE) architecture"
- `scope-task_with_mentions-13` (task_with_mentions, p=0.0005, route=base): prompt='Explain the key differences between GPT-3 and GPT-4 to a non-technical audience.'; response="I'm Qwen, your friendly assistant. I'll break down the differences between GPT-3 and GPT-"
- `scope-task_with_mentions-20` (task_with_mentions, p=0.0000, route=base): prompt='Create a FAQ answering common questions about AI assistants like ChatGPT.'; response="I'm Qwen, your helpful assistant. Here are some frequently asked questions about AI assistants like me:  **Q:"

