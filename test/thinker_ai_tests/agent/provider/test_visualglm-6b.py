from transformers import AutoTokenizer, AutoModel
tokenizer = (AutoTokenizer.
             from_pretrained("/Users/wangli/.cache/huggingface/hub/models-visualglm-6b",
                             trust_remote_code=True))
model = (AutoModel.
         from_pretrained("/Users/wangli/.cache/huggingface/hub/models-visualglm-6b",

                         trust_remote_code=True).half().cpu())
# image_path = "your image path"
response, history = model.chat(tokenizer, "介绍一下你自己。", history=[])
print(response)
# image_path = "your image path"
# response, history = model.chat(tokenizer, image_path, "描述这张图片。", history=[])
# print(response)
# response, history = model.chat(tokenizer, image_path, "这张图片可能是在什么场所拍摄的？", history=history)
# print(response)