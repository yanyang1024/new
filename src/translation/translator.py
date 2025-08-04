"""
翻译器模块
包含翻译处理的核心逻辑
"""

import numpy as np
import torch
import queue
import threading
import time
from datetime import datetime
from typing import Optional, List, Tuple
import logging

from transformers import AutoProcessor, SeamlessM4Tv2ForSpeechToText
from ..config.audio_config import RATE, PATH_FOR_SEAMLESS_M4T
from ..config.translation_config import (
    MAX_NEW_TOKENS, NUM_BEAMS, EARLY_STOPPING,
    DEFAULT_TARGET_LANG, MAX_TRANSLATION_HISTORY
)
from ..utils.translation_utils import (
    format_translation, should_merge_translations, 
    is_complete_sentence, save_translation_log,
    truncate_translation_history, truncate_previous_translations
)
from ..utils.audio_utils import save_test_audio, save_test_translation

logger = logging.getLogger(__name__)

class Translator:
    """翻译器类"""
    
    def __init__(self, test_mode: bool = True):
        """
        初始化翻译器
        
        Args:
            test_mode: 是否为测试模式
        """
        self.test_mode = test_mode
        
        # 设置设备
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # 初始化模型
        self._setup_translator()
        
        # 翻译状态
        self.translation_queue = queue.Queue()
        self.current_translation = ""
        self.partial_translation = ""
        self.latest_complete_translation = ""
        self.all_translations = []
        self.translation_history = []
        
        # 翻译处理相关
        self.previous_translations = []
        self.last_chinese_text = ""
        self.last_sentence_end = time.time()
        self.latest_partial = ""
        self.last_update_time = 0
        
        # 线程控制
        self.running = False
        
        # 启动翻译处理线程
        self.running = True
        self.translation_thread = threading.Thread(target=self._translation_worker)
        self.translation_thread.start()
    
    def _setup_translator(self):
        """设置翻译器"""
        try:
            import transformers
            if transformers.__version__ < "4.37.2":
                logger.warning("Warning: Please upgrade transformers to >= 4.37.2")
        except ImportError:
            raise ImportError("Please install transformers: pip install transformers")
        
        # 设置GPU性能优化
        torch.backends.cudnn.benchmark = True
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        logger.info(f"[INFO] 使用设备: {self.device}")
        logger.info(f"[INFO] 当前CUDA设备: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")
        
        # 加载模型和处理器
        logger.info("[INFO] 开始加载模型...")
        self.processor = AutoProcessor.from_pretrained(PATH_FOR_SEAMLESS_M4T, device_map="auto")
        
        self.model = SeamlessM4Tv2ForSpeechToText.from_pretrained(
            PATH_FOR_SEAMLESS_M4T,
            device_map="auto",
            torch_dtype=torch.float16
        ).to(self.device).eval()
        logger.info("[INFO] 模型加载完成")
    
    def process_speech_segment(self, audio_data: bytes, partial: bool = False):
        """
        处理语音片段
        
        Args:
            audio_data: 音频数据
            partial: 是否为部分处理
        """
        try:
            if self.test_mode:
                self._append_debug("开始处理语音片段" + (" (部分识别)" if partial else ""))

            # 将原始字节数据转换为 numpy 数组
            if isinstance(audio_data, bytes):
                audio_np = np.frombuffer(audio_data, dtype=np.float32)
                if self.test_mode:
                    self._append_debug(f"音频数据转换: bytes -> numpy, 长度={len(audio_np)}")
            else:
                audio_np = audio_data
                if self.test_mode:
                    self._append_debug(f"音频数据已经是numpy格式, 长度={len(audio_np)}")

            # 检查音频数据是否有效
            if len(audio_np) == 0:
                if self.test_mode:
                    self._append_debug("音频数据为空，跳过处理")
                return

            # 标准化音频数据（确保在-1 到1 之间）
            max_abs = np.max(np.abs(audio_np))
            if max_abs > 0:
                audio_np = audio_np / max_abs
                if self.test_mode:
                    self._append_debug(f"音频数据标准化: 最大值={max_abs:.4f} -> 1.0")
            else:
                if self.test_mode:
                    self._append_debug("音频数据为静音，跳过标准化")

            # 检查标准化后的数据
            if np.isnan(audio_np).any() or np.isinf(audio_np).any():
                if self.test_mode:
                    self._append_debug("标准化后音频数据包含无效值，跳过处理")
                return

            try:
                if self.test_mode:
                    self._append_debug("开始SeamlessM4T处理...")
                
                # 使用 SeamlessM4T 进行端到端处理（语音->英文文本）
                audio_inputs = self.processor(
                    audios=audio_np,
                    return_tensors="pt",
                    padding=True,
                    sampling_rate=RATE
                ).to(self.device) 

                if self.test_mode:
                    self._append_debug("音频预处理完成，开始生成翻译...")

                # 生成翻译（直接输出英文）
                with torch.inference_mode(), torch.cuda.amp.autocast():
                    output_tokens = self.model.generate(
                        **audio_inputs,
                        tgt_lang=DEFAULT_TARGET_LANG,
                        max_new_tokens=MAX_NEW_TOKENS,
                        num_beams=NUM_BEAMS,
                        early_stopping=EARLY_STOPPING
                    )

                if self.test_mode:
                    self._append_debug("翻译生成完成，开始解码...")

                # 解码结果
                translated_text = self.processor.decode(
                    output_tokens[0].cpu().numpy().squeeze(), 
                    skip_special_tokens=True
                )

                if self.test_mode:
                    self._append_debug(f"识别翻译结果: {translated_text}")

                # 检查翻译结果是否有效
                if not translated_text or translated_text.strip() == "":
                    if self.test_mode:
                        self._append_debug("翻译结果为空，跳过处理")
                    return

                # 处理部分结果，翻译结果入队
                if partial: 
                    # 部分结果需要进行部分处理 - 合并与更新翻译文本
                    if self.last_chinese_text and translated_text.startswith(self.last_chinese_text):
                        translated_text = translated_text[len(self.last_chinese_text):].strip()
                        if self.test_mode:
                            self._append_debug(f"部分翻译去重: {translated_text}")
                    self.last_chinese_text = translated_text
                    self.translation_queue.put((translated_text, True))
                    if self.test_mode:
                        self._append_debug("部分翻译结果已入队")
                else:
                    self.translation_queue.put((translated_text, False))
                    if self.test_mode:
                        self._append_debug("完整翻译结果已入队")

            except Exception as e:
                if self.test_mode:
                    self._append_debug(f"SeamlessM4T 处理异常: {str(e)}")
                translated_text = f"Translation Error: {str(e)}"

            # 处理临时文件（仅测试模式保留）
            if self.test_mode:
                save_test_audio(audio_np, self.test_mode)
                save_test_translation(translated_text, self.test_mode)

        except Exception as e:
            if self.test_mode:
                self._append_debug(f"语音分段处理异常: {e}")
            translated_text = f"System Error: {str(e)}"
    
    def _translation_worker(self):
        """翻译处理工作线程"""
        while self.running:
            try:
                chinese_text, is_partial = self.translation_queue.get(timeout=1)
                if not self.running:
                    break
                
                if chinese_text:
                    if self.test_mode:
                        self._append_debug(f"翻译中: {chinese_text} ({'部分' if is_partial else '完整'})")
                    
                    # 确定是否需要合并翻译
                    should_merge = should_merge_translations(
                        chinese_text, 
                        self.partial_translation if is_partial else self.current_translation,
                        self.last_sentence_end
                    )
                    
                    if should_merge and self.test_mode:
                        self._append_debug("检测到语义连贯，将合并翻译")

                    translated_text = chinese_text
                    
                    # 判断句子完整性
                    is_complete = is_complete_sentence(translated_text)
                    if is_complete and is_partial:
                        # 如果检测到句子完整，更新部分翻译标志
                        is_partial = False
                        if self.test_mode:
                            self._append_debug("检测到完整句子，更新为完整翻译")
                    
                    if is_partial:  # 处理部分翻译
                        if should_merge and self.partial_translation:
                            self.partial_translation = f"{self.partial_translation} {translated_text}"
                            if self.test_mode:
                                self._append_debug(f"合并部分翻译: {self.partial_translation}")
                        else:
                            self.partial_translation = translated_text
                            if self.test_mode:
                                self._append_debug(f"设置部分翻译: {self.partial_translation}")
                        self.current_translation = f"{self.partial_translation}..."

                        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        self.translation_history.append(f"[{timestamp}] (部分) {translated_text}")

                    else:  # 处理完整翻译
                        formatted_text = format_translation(translated_text, False)
                        if should_merge and self.current_translation:
                            merged_text = f"{self.current_translation.rstrip('.')} {formatted_text}"
                            self.current_translation = merged_text
                            if self.test_mode:
                                self._append_debug(f"合并完整翻译: {merged_text}")
                        else:
                            self.previous_translations.append(self.current_translation)
                            self.previous_translations = truncate_previous_translations(self.previous_translations)
                            self.current_translation = formatted_text
                            if self.test_mode:
                                self._append_debug(f"设置完整翻译: {formatted_text}")
                        self.partial_translation = ""

                        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        self.translation_history.append(f"[{timestamp}] {formatted_text}")
                        self.latest_partial = formatted_text
                        self.last_update_time = time.time()
                        
                        # 更新最新完整翻译和所有翻译记录
                        self.latest_complete_translation = formatted_text
                        self.all_translations.append(formatted_text)
                        self.all_translations = truncate_translation_history(self.all_translations)
                        
                        # 保存完整翻译到日志
                        save_translation_log(formatted_text, False)
                        
                        if self.test_mode:
                            self._append_debug(f"完整翻译已保存: {formatted_text}")
                    
                    if self.test_mode:
                        self._append_debug(f"翻译结果: {self.current_translation} ({'部分' if is_partial else '完整'})")
                        
            except queue.Empty:
                continue
            except Exception as e:
                if self.test_mode:
                    self._append_debug(f"翻译异常: {str(e)}")
                self.current_translation = f"翻译出错: {str(e)}"
    
    def _append_debug(self, msg: str):
        """添加调试信息"""
        if not self.test_mode:
            return
            
        timestamp = time.strftime('%H:%M:%S')
        self.debug_msgs.append(f"[{timestamp}] {msg}")
        
        # 限制debug信息长度
        if len(self.debug_msgs) > 30:
            self.debug_msgs = self.debug_msgs[-30:]
    
    def get_debug_info(self) -> str:
        """获取调试信息"""
        return '\n'.join(getattr(self, 'debug_msgs', []))
    
    def get_all_translations(self) -> str:
        """获取所有翻译记录的汇总"""
        return "\n".join(self.all_translations)
    
    def reset_state(self):
        """重置翻译状态"""
        self.current_translation = ""
        self.partial_translation = ""
        self.latest_complete_translation = ""
        self.all_translations = []
        self.translation_history = []
        self.previous_translations = []
        self.last_chinese_text = ""
        self.latest_partial = ""
        self.last_update_time = 0
        self.debug_msgs = []
        
        # 清空翻译队列
        while not self.translation_queue.empty():
            try:
                self.translation_queue.get_nowait()
            except queue.Empty:
                break
    
    def close(self):
        """关闭翻译器"""
        self.running = False
        
        # 停止翻译线程
        if hasattr(self, 'translation_thread') and self.translation_thread.is_alive():
            logger.info("[INFO] 停止翻译线程...")
            self.translation_queue.put(("", False))  # 发送空消息触发退出
            self.translation_thread.join(timeout=5)
            if self.translation_thread.is_alive():
                logger.warning("[WARNING] 翻译线程未能在5秒内停止") 