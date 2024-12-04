import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from moviepy.config import change_settings
change_settings({"IMAGEMAGICK_BINARY": "/opt/homebrew/bin/convert"})  # Update the path accordingly

import unittest
import os
import shutil
import torch
import faulthandler
from thinker_ai.agent.tools.text_to_video_generator import TextToVideoGenerator
from thinker_ai.configs.const import TEST_DATA_PATH


class TestTextToVideoGenerator(unittest.TestCase):
    def setUp(self):
        faulthandler.enable()
        # 设置测试环境
        self.template_path = os.path.join(TEST_DATA_PATH, "test_template.png")
        self.image_library_dir = os.path.join(TEST_DATA_PATH, "test_images")
        self.output_video_path = os.path.join(TEST_DATA_PATH, "test_output_video.mp4")
        self.font_path = None  # 不指定字体，使用系统默认字体
        self.text_input = "测试文本"

        # 创建测试图片库和模板
        os.makedirs(self.image_library_dir, exist_ok=True)
        self._create_test_images()
        self._create_test_template()

        # 创建生成器实例，强制使用 CPU
        self.generator = TextToVideoGenerator(
            template_path=self.template_path,
            image_library_dir=self.image_library_dir,
            output_video_path=self.output_video_path,
            font_path=self.font_path,
            device='cpu'
        )
        # 获取模型和处理器
        self.model = self.generator.model
        self.processor = self.generator.processor

    def tearDown(self):
        # 清理测试环境
        if os.path.exists(self.image_library_dir):
            shutil.rmtree(self.image_library_dir)
        if os.path.exists(self.template_path):
            os.remove(self.template_path)
        if os.path.exists(self.output_video_path):
            os.remove(self.output_video_path)
        features_file = os.path.join(self.image_library_dir, 'image_features.json')
        if os.path.exists(features_file):
            os.remove(features_file)

    def _create_test_images(self):
        # 创建测试图片
        from PIL import Image
        for i in range(3):
            image = Image.new('RGB', (100, 100), color=(i*80, i*80, i*80))
            image_path = os.path.join(self.image_library_dir, f"test_image_{i}.jpg")
            image.save(image_path)

    def _create_test_template(self):
        # 创建测试模板
        from PIL import Image
        template = Image.new('RGBA', (500, 500), color=(255, 255, 255, 255))
        template.save(self.template_path)

    def test_model_and_processor_loading(self):
        # 测试模型和处理器加载
        try:
            self.assertIsNotNone(self.model)
            self.assertIsNotNone(self.processor)
            print("模型和处理器加载成功。")
        except Exception as e:
            self.fail(f"模型或处理器加载失败: {e}")

    def test_library_versions(self):
        # 打印库的版本信息
        import transformers
        print(f"PyTorch 版本: {torch.__version__}")
        print(f"Transformers 版本: {transformers.__version__}")

    def test_processor_text_input(self):
        # 测试处理器处理文本输入
        try:
            text = "测试文本"
            inputs = self.processor(text=[text], return_tensors="pt", padding=True)
            self.assertIn('input_ids', inputs)
            self.assertIn('attention_mask', inputs)
            print("处理器成功处理文本输入。")
        except Exception as e:
            self.fail(f"处理器处理文本输入失败: {e}")

    def test_model_get_text_features(self):
        # 测试获取文本特征
        try:
            text = "测试文本"
            inputs = self.processor(text=[text], return_tensors="pt", padding=True)
            inputs = {k: v.to(self.generator.device) for k, v in inputs.items()}
            print("输入张量信息：")
            for k, v in inputs.items():
                print(f"{k}: shape={v.shape}, dtype={v.dtype}, device={v.device}")
            with torch.no_grad():
                text_features = self.model.get_text_features(**inputs)
            self.assertIsNotNone(text_features)
            print("成功获取文本特征。")
        except Exception as e:
            self.fail(f"获取文本特征失败: {e}")

    def test_model_get_text_features_simple_text(self):
        # 测试获取简单文本的文本特征
        try:
            text = "Hello world"
            inputs = self.processor(text=[text], return_tensors="pt", padding=True)
            inputs = {k: v.to(self.generator.device) for k, v in inputs.items()}
            print("输入张量信息（简单文本）：")
            for k, v in inputs.items():
                print(f"{k}: shape={v.shape}, dtype={v.dtype}, device={v.device}")
            with torch.no_grad():
                text_features = self.model.get_text_features(**inputs)
            self.assertIsNotNone(text_features)
            print("成功获取简单文本的文本特征。")
        except Exception as e:
            self.fail(f"获取简单文本的文本特征失败: {e}")


    def test_process_video_from_text(self):
        # 测试生成视频的完整流程
        try:
            output_path = self.generator.process_video_from_text(self.text_input)
            self.assertTrue(os.path.exists(output_path))
            print(f"生成的视频文件路径: {output_path}")
        except Exception as e:
            self.fail(f"process_video_from_text 方法执行失败: {e}")

    def test_select_image_by_text(self):
        # 测试图片选择功能
        try:
            selected_image = self.generator._select_image_by_text(self.text_input)
            self.assertTrue(os.path.exists(selected_image))
            print(f"选定的图片文件路径: {selected_image}")
        except Exception as e:
            self.fail(f"_select_image_by_text 方法执行失败: {e}")
    def test_apply_template(self):
        # 测试模板应用功能
        try:
            selected_image = self.generator._select_image_by_text(self.text_input)
            templated_image = self.generator._apply_template(
                image_path=selected_image,
                position=(50, 50),
                resize_to=(200, 200),
                text=self.text_input
            )
            self.assertTrue(os.path.exists(templated_image))
            print(f"模板化的图片文件路径: {templated_image}")
        except Exception as e:
            self.fail(f"_apply_template 方法执行失败: {e}")

    def test_generate_video(self):
        # 测试视频生成功能
        try:
            selected_image = self.generator._select_image_by_text(self.text_input)
            templated_image = self.generator._apply_template(
                image_path=selected_image,
                position=(50, 50),
                resize_to=(200, 200),
                text=self.text_input
            )
            video_path = self.generator._generate_video(
                image_paths=[templated_image],
                fps=1,
                resolution=(640, 480)
            )
            self.assertTrue(os.path.exists(video_path))
            print(f"生成的视频文件路径: {video_path}")
        except Exception as e:
            self.fail(f"_generate_video 方法执行失败: {e}")

    def test_add_subtitle(self):
        # Test subtitle addition functionality
        try:
            from moviepy.editor import ColorClip
            test_video_path = os.path.join(self.generator.temp_dir, "test_video.mp4")
            clip = ColorClip(size=(640, 480), color=(255, 0, 0))
            clip.duration = 2  # 2 seconds
            clip.fps = 24  # Set frames per second
            clip.write_videofile(test_video_path, codec=self.generator.encoder)
            clip.close()

            final_video_path = self.generator._add_subtitle(
                video_path=test_video_path,
                text=self.text_input,
                position=("center", "bottom"),
                font_size=24,
                color="white"
            )
            self.assertTrue(os.path.exists(final_video_path))
            print(f"Added subtitles to video file at: {final_video_path}")
        except Exception as e:
            self.fail(f"_add_subtitle method failed: {e}")


if __name__ == '__main__':
    unittest.main()