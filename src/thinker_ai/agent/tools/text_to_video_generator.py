import os
import uuid
import json
import shutil
import threading
import tempfile
from pathlib import Path
from typing import List, Tuple, Optional, Union

import torch
from PIL import Image, ImageDraw, ImageFont
from moviepy.editor import (
    ImageSequenceClip, TextClip, CompositeVideoClip, VideoFileClip
)
from moviepy.video.fx.resize import resize  # 确保正确导入
from transformers import CLIPProcessor, CLIPModel

from thinker_ai.common.logs import logger


class TextToVideoGenerator:
    _model_lock = threading.Lock()
    _model = None
    _processor = None

    def __init__(self, template_path: str, image_library_dir: str, output_video_path: str,
                 font_path: Optional[str] = None, device: Optional[str] = None, encoder: str = 'libx264'):
        self.template_path = template_path
        self.image_library_dir = image_library_dir
        self.output_video_path = output_video_path
        self.encoder = encoder
        self.temp_dir = tempfile.mkdtemp()
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        # 字体处理，针对 macOS 系统
        if font_path and os.path.exists(font_path):
            self.font_path = font_path
        else:
            # 尝试使用系统默认字体
            possible_fonts = [
                "/System/Library/Fonts/Helvetica.ttc",  # macOS 系统字体
                "/Library/Fonts/Arial.ttf",             # 如果用户已安装 Arial
                "Helvetica",                            # PIL 可能识别的字体名称
                "Arial"
            ]
            self.font_path = None
            for font in possible_fonts:
                try:
                    ImageFont.truetype(font, size=24)
                    self.font_path = font
                    logger.info(f"使用字体: {font}")
                    break
                except IOError:
                    continue
            if not self.font_path:
                logger.warning("未找到可用字体，将使用默认字体。")

        # 加载模型（线程安全）
        with TextToVideoGenerator._model_lock:
            if TextToVideoGenerator._model is None or TextToVideoGenerator._processor is None:
                try:
                    logger.info("加载 CLIP 模型...")
                    TextToVideoGenerator._model = CLIPModel.from_pretrained(
                        "openai/clip-vit-base-patch32").to(self.device)
                    TextToVideoGenerator._processor = CLIPProcessor.from_pretrained(
                        "openai/clip-vit-base-patch32")
                    logger.info("模型加载完成。")
                except Exception as e:
                    logger.error(f"模型加载失败: {e}")
                    raise RuntimeError(f"模型加载失败: {e}")
        self.model = TextToVideoGenerator._model
        self.processor = TextToVideoGenerator._processor

        # 加载或生成图片特征向量
        self.features_file = os.path.join(self.image_library_dir, 'image_features.json')
        self.image_features = self._load_or_compute_image_features()

    def _load_or_compute_image_features(self):
        """加载或计算图片特征向量"""
        logger.info("开始加载或计算图片特征向量...")
        if os.path.exists(self.features_file):
            logger.info("加载预计算的图片特征向量...")
            with open(self.features_file, 'r') as f:
                image_features = json.load(f)
            # 将特征转换为 Tensor
            for key in image_features:
                image_features[key] = torch.tensor(image_features[key]).to(self.device)
        else:
            logger.info("计算图片特征向量...")
            image_features = {}
            images = list(Path(self.image_library_dir).glob("*.jpg"))
            if not images:
                raise FileNotFoundError("图片素材库中没有找到任何 JPG 图片。")
            for img_path in images:
                try:
                    logger.info(f"正在处理图片: {img_path}")
                    image = Image.open(str(img_path)).convert("RGB")
                    inputs = self.processor(images=image, return_tensors="pt")
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}  # 修正此处
                    with torch.no_grad():
                        features = self.model.get_image_features(**inputs)
                        features = features.cpu().numpy().tolist()[0]
                        image_features[str(img_path)] = features
                except Exception as e:
                    logger.error(f"处理图片 {img_path} 时发生错误: {e}")
                    continue
            # 保存特征向量到文件
            with open(self.features_file, 'w') as f:
                json.dump(image_features, f)
            # 将特征转换为 Tensor
            for key in image_features:
                image_features[key] = torch.tensor(image_features[key]).to(self.device)
        return image_features

    def _select_image_by_text(self, text: str) -> str:
        """选择与文本最匹配的图片素材（私有方法）"""
        if not self.image_features:
            raise RuntimeError("没有可用的图片特征向量。")

        # 计算文本特征向量
        inputs = self.processor(text=[text], return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}  # 修正此处
        with torch.no_grad():
            text_features = self.model.get_text_features(**inputs)
        text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)

        # 计算余弦相似度并选择最佳匹配
        max_sim = -1
        best_image_path = None
        for img_path, img_feature in self.image_features.items():
            img_feature = img_feature / img_feature.norm(p=2, dim=-1, keepdim=True)
            sim = torch.sum(text_features * img_feature)
            if sim > max_sim:
                max_sim = sim
                best_image_path = img_path
        if best_image_path is None:
            raise RuntimeError("未能找到匹配的图片。")
        return best_image_path

    def _apply_template(self, image_path: str, position: Tuple[int, int],
                        resize_to: Tuple[int, int], text: Optional[str] = None) -> str:
        """将图片嵌入模板并保存（私有方法）"""
        try:
            template = Image.open(self.template_path).convert("RGBA")
            image = Image.open(image_path).convert("RGBA")
        except Exception as e:
            logger.error(f"无法打开模板或图片文件: {e}")
            raise FileNotFoundError(f"无法打开模板或图片文件: {e}")

        image = image.resize(resize_to)
        template.paste(image, position, image)

        if text:
            draw = ImageDraw.Draw(template)
            try:
                if self.font_path and os.path.exists(self.font_path):
                    font = ImageFont.truetype(self.font_path, 24)
                else:
                    font = ImageFont.load_default()
            except IOError:
                logger.warning("指定的字体文件未找到，使用默认字体。")
                font = ImageFont.load_default()
            draw.text((50, template.height - 40), text, fill="white", font=font)

        output_path = os.path.join(self.temp_dir, f"templated_image_{uuid.uuid4().hex}.png")
        template.save(output_path)
        return output_path

    def _generate_video(self, image_paths: List[str], fps: int = 1,
                        resolution: Tuple[int, int] = (1920, 1080)) -> str:
        """合成视频（私有方法）"""
        clip = ImageSequenceClip(image_paths, fps=fps)
        clip = resize(clip, newsize=resolution)
        clip.write_videofile(self.output_video_path, codec=self.encoder)
        clip.close()
        return self.output_video_path

    def _add_subtitle(self, video_path: str, text: str,
                      position: Union[str, Tuple[Union[int, float, str], Union[int, float, str]]] = ("center", "bottom"),
                      font_size: int = 24, color: str = "white") -> str:
        """为视频添加字幕（私有方法）"""
        try:
            video = VideoFileClip(video_path)
        except Exception as e:
            logger.error(f"无法打开视频文件: {e}")
            raise FileNotFoundError(f"无法打开视频文件: {e}")

        try:
            if self.font_path and os.path.exists(self.font_path):
                txt_clip = TextClip(text, fontsize=font_size, color=color, font=self.font_path)
            else:
                txt_clip = TextClip(text, fontsize=font_size, color=color)
        except Exception as e:
            logger.warning(f"无法加载指定字体，使用默认字体: {e}")
            txt_clip = TextClip(text, fontsize=font_size, color=color)

        txt_clip = txt_clip.set_position(position).set_duration(video.duration)
        final_video = CompositeVideoClip([video, txt_clip])

        final_output_path = os.path.join(self.temp_dir, f"final_output_video_{uuid.uuid4().hex}.mp4")
        final_video.write_videofile(final_output_path, codec=self.encoder)
        video.close()
        txt_clip.close()
        final_video.close()
        return final_output_path

    def process_video_from_text(self, text: str) -> str:
        """一键完成整个视频生成过程，公有方法"""
        logger.info("开始处理视频...")
        try:
            # Step 1: 选择匹配的图片
            logger.info("开始选择匹配的图片...")
            selected_image = self._select_image_by_text(text)
            logger.info(f"已选择图片: {selected_image}")

            # Step 2: 应用模板
            logger.info("应用模板并生成图片...")
            templated_image = self._apply_template(
                selected_image, position=(50, 100), resize_to=(300, 300), text=text)

            # Step 3: 生成视频
            logger.info("生成视频...")
            video_path = self._generate_video([templated_image])

            # Step 4: 添加字幕
            logger.info("添加字幕...")
            final_video_path = self._add_subtitle(video_path, text=text)

            # 移动最终视频到指定输出路径
            shutil.move(final_video_path, self.output_video_path)
            logger.info(f"视频生成完成，输出路径: {self.output_video_path}")
            return self.output_video_path
        except Exception as e:
            logger.error(f"视频生成失败: {e}")
            raise RuntimeError(f"视频生成失败: {e}")
        finally:
            # 清理临时文件夹
            shutil.rmtree(self.temp_dir, ignore_errors=True)

    def __del__(self):
        # 确保在对象销毁时清理临时文件夹
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir, ignore_errors=True)