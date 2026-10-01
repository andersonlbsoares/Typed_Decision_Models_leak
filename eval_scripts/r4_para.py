"""Meaning-preserving paraphrases with Qwen2.5-7B-Instruct (vLLM, greedy)."""
import json, time
from vllm import LLM, SamplingParams
items = json.load(open("r4_to_para.json"))
llm = LLM("Qwen/Qwen2.5-7B-Instruct", max_model_len=2048, gpu_memory_utilization=0.85, max_num_seqs=128)
tok = llm.get_tokenizer()
INSTR = ("Paraphrase the text below. Keep exactly the same meaning, topic, sentiment, emotion and intent, "
         "but use different wording and sentence structure. Keep roughly the same length. "
         "Output only the paraphrase, nothing else.\n\nText:\n")
prompts = [tok.apply_chat_template([{"role": "user", "content": INSTR + x["text"]}], tokenize=False,
                                   add_generation_prompt=True) for x in items]
sp = SamplingParams(temperature=0.0, max_tokens=320)
t = time.time(); outs = llm.generate(prompts, sp)
res = {x["id"]: dict(orig=x["text"], para=o.outputs[0].text.strip()) for x, o in zip(items, outs)}
json.dump(res, open("r4_para.json", "w"))
print("paraphrased", len(res), round(time.time() - t), "s", flush=True)
