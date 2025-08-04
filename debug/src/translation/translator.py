import numpy as np
import torch
import soundfile as sf
import os
import time
from datetime import datetime
from ..config.audio_config import *
from ..config.translation_config import *
from ..utils.translation_utils import *

class TranslationProcessor:
    def __init__(self, translator):
        self.translator = translator
        self.test_mode = translator.test_mode

    def process_speech_segment(self, audio_data, partial=False):
        """处理语音片段进行翻译"""
        try:
            if self.test_mode:
                self.translator.debug_info['recognition_attempts'] += 1
                self.translator._append_debug("开始处理语音片段" + (" (部分识别)" if partial else ""))

            # 将原始字节数据转换为 numpy 数组
            if isinstance(audio_data, bytes):
                audio_np = np.frombuffer(audio_data, dtype=np.float32)
                if self.test_mode:
                    self.translator._append_debug(f"音频数据转换: bytes -> numpy, 长度={len(audio_np)}")
            else:
                audio_np = audio_data
                if self.test_mode:
                    self.translator._append_debug(f"音频数据已经是numpy格式, 长度={len(audio_np)}")

            # 检查音频数据是否有效
            if len(audio_np) == 0:
                if self.test_mode:
                    self.translator._append_debug("音频数据为空，跳过处理")
                return

            # 标准化音频数据（确保在-1 到1 之间）
            max_abs = np.max(np.abs(audio_np))
            if max_abs > 0:
                audio_np = audio_np / max_abs
                if self.test_mode:
                    self.translator._append_debug(f"音频数据标准化: 最大值={max_abs:.4f} -> 1.0")
            else:
                if self.test_mode:
                    self.translator._append_debug("音频数据为静音，跳过标准化")

            # 检查标准化后的数据
            if np.isnan(audio_np).any() or np.isinf(audio_np).any():
                if self.test_mode:
                    self.translator._append_debug("标准化后音频数据包含无效值，跳过处理")
                return

            try:
                if self.test_mode:
                    self.translator._append_debug("开始SeamlessM4T处理...")
                
                # 使用 SeamlessM4T 进行端到端处理（语音->英文文本）
                audio_inputs = self.translator.processor(
                    audios=audio_np,
                    return_tensors="pt",
                    padding=True,
                    sampling_rate=16000
                ).to(self.translator.device) 

                if self.test_mode:
                    self.translator._append_debug("音频预处理完成，开始生成翻译...")

                # 生成翻译（直接输出英文）
                with torch.inference_mode(), torch.cuda.amp.autocast():
                    output_tokens = self.translator.model.generate(
                        **audio_inputs,
                        tgt_lang=TARGET_LANGUAGE,
                        max_new_tokens=MODEL_CONFIG['max_new_tokens'],
                        num_beams=MODEL_CONFIG['num_beams'],
                        early_stopping=MODEL_CONFIG['early_stopping']
                    )

                if self.test_mode:
                    self.translator._append_debug("翻译生成完成，开始解码...")

                # 解码结果
                translated_text = self.translator.processor.decode(
                    output_tokens[0].cpu().numpy().squeeze(), 
                    skip_special_tokens=True
                )

                if self.test_mode:
                    self.translator.debug_info['recognition_success'] += 1
                    self.translator._append_debug(f"识别翻译结果: {translated_text}")

                # 检查翻译结果是否有效
                if not validate_translation_result(translated_text, self.test_mode):
                    if self.test_mode:
                        self.translator._append_debug("翻译结果无效，跳过处理")
                    return

                # 处理部分结果，翻译结果入队 结果包含翻译文本和是否为完整结果的flag,True表示为部分结果，False表示为完整结果
                if partial: 
                    # 部分结果需要进行部分处理 - 合并与更新翻译文本
                    # 将self.last_chinese_text 与 translated_text 进行比较，如果 translated_text 以 self.last_chinese_text 开头，则删除 self.last_chinese_text 的部分内容，并更新 self.last_chinese_text。
                    if self.translator.last_chinese_text and translated_text.startswith(self.translator.last_chinese_text):
                        translated_text = translated_text[len(self.translator.last_chinese_text):].strip()
                        if self.test_mode:
                            self.translator._append_debug(f"部分翻译去重: {translated_text}")
                    self.translator.last_chinese_text = translated_text
                    self.translator.translation_queue.put((translated_text, True))
                    if self.test_mode:
                        self.translator._append_debug("部分翻译结果已入队")
                else:
                    self.translator.translation_queue.put((translated_text, False))
                    if self.test_mode:
                        self.translator._append_debug("完整翻译结果已入队")

            except Exception as e:
                if self.test_mode:
                    self.translator._append_debug(f"SeamlessM4T 处理异常: {str(e)}")
                translated_text = f"Translation Error: {str(e)}"

            # 处理临时文件（仅测试模式保留）
            if self.test_mode:
                wav_path = f"test_segments/segment_{int(time.time())}.wav"
                os.makedirs(os.path.dirname(wav_path), exist_ok=True)
                sf.write(wav_path, audio_np, RATE)
                if self.test_mode:
                    self.translator._append_debug(f"保存测试音频片段: {wav_path}")

                translated_text_path = f"test_segments_translated/segment_{int(time.time())}.txt"
                os.makedirs(os.path.dirname(translated_text_path), exist_ok=True)
                with open(translated_text_path, 'w', encoding='utf-8') as f:
                    f.write(translated_text)

        except Exception as e:
            if self.test_mode:
                self.translator._append_debug(f"语音分段处理异常: {e}")
            translated_text = f"System Error: {str(e)}"

    def _translation_worker(self):
        """翻译工作线程"""
        while self.translator.running:
            try:
                chinese_text, is_partial = self.translator.translation_queue.get(timeout=1)
                if not self.translator.running:
                    break
                
                if chinese_text:
                    if self.test_mode:
                        self.translator.debug_info['translation_attempts'] += 1
                        self.translator._append_debug(f"翻译中: {chinese_text} ({'部分' if is_partial else '完整'})")
                    
                    # 确定是否需要合并翻译
                    should_merge = should_merge_translations(
                        chinese_text, 
                        self.translator.partial_translation if is_partial else self.translator.current_translation,
                        self.translator.last_sentence_end,
                        self.test_mode
                    )
                    
                    if should_merge and self.test_mode:
                        self.translator._append_debug("检测到语义连贯，将合并翻译")

                    translated_text = chinese_text
                    
                    # 判断句子完整性
                    is_complete = is_complete_sentence(translated_text)
                    if is_complete and is_partial:
                        # 如果检测到句子完整，更新部分翻译标志
                        is_partial = False
                        if self.test_mode:
                            self.translator._append_debug("检测到完整句子，更新为完整翻译")
                    
                    if is_partial:  # 处理部分翻译
                        if should_merge and self.translator.partial_translation:
                            self.translator.partial_translation = f"{self.translator.partial_translation} {translated_text}"
                            if self.test_mode:
                                self.translator._append_debug(f"合并部分翻译: {self.translator.partial_translation}")
                        else:
                            self.translator.partial_translation = translated_text
                            if self.test_mode:
                                self.translator._append_debug(f"设置部分翻译: {self.translator.partial_translation}")
                        self.translator.current_translation = f"{self.translator.partial_translation}..."

                        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        self.translator.translation_history.append(f"[{timestamp}] (部分) {translated_text}")

                    else:  # 处理完整翻译
                        formatted_text = format_translation(translated_text, False)
                        if should_merge and self.translator.current_translation:
                            merged_text = f"{self.translator.current_translation.rstrip('.')} {formatted_text}"
                            self.translator.current_translation = merged_text
                            if self.test_mode:
                                self.translator._append_debug(f"合并完整翻译: {merged_text}")
                        else:
                            self.translator.previous_translations.append(self.translator.current_translation)
                            self.translator.previous_translations = self.translator.previous_translations[-5:]
                            self.translator.current_translation = formatted_text
                            if self.test_mode:
                                self.translator._append_debug(f"设置完整翻译: {formatted_text}")
                        self.translator.partial_translation = ""

                        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        self.translator.translation_history.append(f"[{timestamp}] {formatted_text}")
                        self.translator.latest_partial = formatted_text
                        self.translator.last_update_time = time.time()
                        
                        # 更新最新完整翻译和所有翻译记录
                        self.translator.latest_complete_translation = formatted_text
                        self.translator.all_translations.append(formatted_text)
                        self.translator.all_translations = self.translator.all_translations[-MAX_HISTORY_LENGTH:]  # 最多保留20条
                        
                        # 保存完整翻译到日志
                        save_translation_log(formatted_text, False, self.test_mode)
                        
                        if self.test_mode:
                            self.translator._append_debug(f"完整翻译已保存: {formatted_text}")
                    
                    if self.test_mode:
                        self.translator._append_debug(f"翻译结果: {self.translator.current_translation} ({'部分' if is_partial else '完整'})")
                        
            except queue.Empty:
                continue
            except Exception as e:
                if self.test_mode:
                    self.translator._append_debug(f"翻译异常: {str(e)}")
                self.translator.current_translation = f"翻译出错: {str(e)}" 