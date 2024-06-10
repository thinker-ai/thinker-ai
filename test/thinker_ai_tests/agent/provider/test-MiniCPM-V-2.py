import unittest

import asynctest
import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer


class TestGPT(asynctest.TestCase):
    def test_ask_for_ocr(self):
        image = Image.open('data/hk_OCR.jpg').convert('RGB')
        model = AutoModel.from_pretrained('/Users/wangli/.cache/huggingface/hub/models--openbmb--MiniCPM-V-2',
                                          trust_remote_code=True, torch_dtype=torch.bfloat16)
        model = model.to(device='mps', dtype=torch.float16)

        tokenizer = AutoTokenizer.from_pretrained('/Users/wangli/.cache/huggingface/hub/models--openbmb--MiniCPM-V-2',
                                                  trust_remote_code=True)
        model.eval()

        question = 'Where is this photo taken?'
        msgs = [{'role': 'user', 'content': question}]

        answer, context, _ = model.chat(
            image=image,
            msgs=msgs,
            context=None,
            tokenizer=tokenizer,
            sampling=True
        )
        print(answer)


if __name__ == '__main__':
    unittest.main()
