from transformers import Qwen3VLMoeForConditionalGeneration, AutoProcessor
import torch

# 加载模型（自动分配设备）
model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
    "Qwen/Qwen3-VL-30B-A3B-Instruct",
    dtype="auto",
    device_map="auto"
)
processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-30B-A3B-Instruct")

# === 输入示例 ===
# 你可以换成本地路径，比如 "test.jpg"
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"
            },
            {"type": "text", "text": "Describe this image in detail."}
        ],
    }
]

# === 预处理 ===
inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt"
)

# === 推理 ===
with torch.no_grad():
    generated_ids = model.generate(**inputs, max_new_tokens=128)

# 去除输入tokens
generated_ids_trimmed = [
    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]

# === 解码输出 ===
output_text = processor.batch_decode(
    generated_ids_trimmed,
    skip_special_tokens=True,
    clean_up_tokenization_spaces=False
)
print("🟢 模型输出:\n", output_text[0])
