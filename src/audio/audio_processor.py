"""
音频处理器模块
包含音频处理的核心逻辑
"""

import numpy as np
import pyaudio
import webrtcvad
import time
from collections import deque
from typing import Optional, List, Tuple
import logging

from ..config.audio_config import (
    RATE, CHANNELS, CHUNK, FORMAT, VAD_MODE,
    PARTIAL_UPDATE_FRAMES, SILENCE_THRESHOLD_SHORT, 
    SILENCE_THRESHOLD_LONG, MIN_SPEECH_FRAMES,
    ENERGY_THRESHOLD, MAX_CACHE_SIZE, CACHE_CLEANUP_INTERVAL
)
from ..utils.audio_utils import normalize_audio_data, convert_audio_to_bytes

logger = logging.getLogger(__name__)

class AudioProcessor:
    """音频处理器类"""
    
    def __init__(self, test_mode: bool = True):
        """
        初始化音频处理器
        
        Args:
            test_mode: 是否为测试模式
        """
        self.test_mode = test_mode
        
        # 初始化音频设备
        self.audio = pyaudio.PyAudio()
        self.vad = webrtcvad.Vad(VAD_MODE)
        
        # 音频处理状态
        self.is_speaking = False
        self.silence_frames = 0
        self.speech_frames = []
        self.consecutive_silence = 0
        self.last_speech_time = time.time()
        
        # 音频缓存
        self.audio_buffer = deque(maxlen=50)  # 约1.5秒的音频
        self.audio_cache = deque(maxlen=MAX_CACHE_SIZE)
        self.frame_counter = 0
        
        # VAD相关
        self.vad_window = deque(maxlen=5)  # VAD结果的滑动窗口
        self.energy_threshold = ENERGY_THRESHOLD
        
        # 调试信息
        self.debug_info = {
            'vad_detections': 0,
            'speech_segments': 0,
            'recognition_attempts': 0,
            'recognition_success': 0,
            'translation_attempts': 0
        }
        
        # 调试消息
        self.debug_msgs = []
        
    def _append_debug(self, msg: str):
        """添加调试信息"""
        if not self.test_mode:
            return
            
        timestamp = time.strftime('%H:%M:%S')
        self.debug_msgs.append(f"[{timestamp}] {msg}")
        
        # 限制debug信息长度
        if len(self.debug_msgs) > 30:
            self.debug_msgs = self.debug_msgs[-30:]
    
    def _is_speech(self, audio_chunk) -> bool:
        """
        检测音频块是否为语音
        
        Args:
            audio_chunk: 音频数据块
            
        Returns:
            是否为语音
        """
        try:
            # 统一将数据转换为float32格式
            if isinstance(audio_chunk, bytes):
                audio_data = np.frombuffer(audio_chunk, dtype=np.float32)
            else:
                audio_data = audio_chunk
            
            # 检查音频数据是否有效
            if audio_data is None or len(audio_data) == 0:
                if self.test_mode:
                    self._append_debug("音频数据为空，跳过VAD检测")
                return False
            
            # 确保数据类型为float32
            audio_data = audio_data.astype(np.float32)
            
            # 处理异常数据范围 - 改进的NaN值处理
            max_abs = np.max(np.abs(audio_data))
            if max_abs > 1000.0:  # 如果数据范围过大，进行软裁剪
                if self.test_mode:
                    self._append_debug(f"音频数据范围异常: {max_abs:.2f}，进行软裁剪")
                # 软裁剪到合理范围
                audio_data = np.clip(audio_data, -1000.0, 1000.0)
                # 重新归一化
                max_abs = np.max(np.abs(audio_data))
                if max_abs > 0:
                    audio_data = audio_data / max_abs
            
            # 检查数据是否包含无效值
            if np.isnan(audio_data).any():
                if self.test_mode:
                    self._append_debug(f"音频数据包含NaN值，尝试修复。数据范围: [{np.min(audio_data):.4f}, {np.max(audio_data):.4f}]")
                # 尝试修复NaN值
                audio_data = np.nan_to_num(audio_data, nan=0.0, posinf=1.0, neginf=-1.0)
            
            if np.isinf(audio_data).any():
                if self.test_mode:
                    self._append_debug(f"音频数据包含Inf值，尝试修复。数据范围: [{np.min(audio_data):.4f}, {np.max(audio_data):.4f}]")
                # 尝试修复Inf值
                audio_data = np.nan_to_num(audio_data, nan=0.0, posinf=1.0, neginf=-1.0)
            
            # 如果是不完整的帧，用0填充到标准长度
            if len(audio_data) < CHUNK:
                pad_length = CHUNK - len(audio_data)
                audio_data = np.pad(audio_data, (0, pad_length), 'constant', constant_values=0)
            elif len(audio_data) > CHUNK:
                audio_data = audio_data[:CHUNK]
            
            # 最终检查处理后的数据
            if np.isnan(audio_data).any() or np.isinf(audio_data).any():
                if self.test_mode:
                    self._append_debug("音频数据处理后仍包含无效值，跳过VAD检测")
                return False
            
            # 改进的音频能量计算 - 防止overflow
            try:
                # 使用更安全的能量计算方法
                abs_audio = np.abs(audio_data)
                # 检查是否有异常大的值
                if np.max(abs_audio) > 1e6:  # 如果最大值超过1e6，说明有异常
                    if self.test_mode:
                        self._append_debug(f"音频数据异常大: {np.max(abs_audio):.2e}，进行裁剪")
                    abs_audio = np.clip(abs_audio, 0, 1e6)
                
                # 使用更稳定的平均值计算
                energy = np.mean(abs_audio)
                
                # 检查能量值是否合理
                if np.isnan(energy) or np.isinf(energy):
                    if self.test_mode:
                        self._append_debug(f"能量计算异常: {energy}，使用默认值")
                    energy = 0.0
                
            except Exception as e:
                if self.test_mode:
                    self._append_debug(f"能量计算异常: {str(e)}，使用默认值")
                energy = 0.0
            
            # 动态调整能量阈值
            if energy > self.energy_threshold * 2:
                self.energy_threshold = min(self.energy_threshold * 1.1, 0.02)
            elif energy < self.energy_threshold / 2:
                self.energy_threshold = max(self.energy_threshold * 0.9, 0.005)
            
            # 准备VAD检测的数据
            # 确保音频数据在合理范围内
            audio_data = np.clip(audio_data, -1.0, 1.0)
            
            # 最终检查
            if np.isnan(audio_data).any() or np.isinf(audio_data).any():
                if self.test_mode:
                    self._append_debug("音频数据裁剪后仍包含无效值，跳过VAD检测")
                return False
            
            vad_data = (audio_data * 32768.0).astype(np.int16)
            
            # 检查转换是否成功
            if np.isnan(vad_data).any() or np.isinf(vad_data).any():
                if self.test_mode:
                    self._append_debug("VAD数据转换失败，包含无效值")
                return False
            
            frame_bytes = vad_data.tobytes()
            
            if len(frame_bytes) % 2 != 0:
                if self.test_mode:
                    self._append_debug("跳过VAD检测：帧字节数不是2的倍数")
                return False
            
            # VAD检测
            vad_result = self.vad.is_speech(frame_bytes, RATE)
            self.vad_window.append(vad_result)
            
            # 使用滑动窗口平均值进行平滑
            vad_ratio = sum(self.vad_window) / len(self.vad_window) if self.vad_window else 0
            
            # 综合判断是否为语音
            is_speech = (energy > self.energy_threshold and vad_ratio > 0.4)
            
            if self.test_mode:
                self.debug_info['vad_detections'] += 1
                self._append_debug(f"VAD检测: {'语音' if is_speech else '静音'} "
                                f"(能量={energy:.4f}, 阈值={self.energy_threshold:.4f}, "
                                f"VAD比例={vad_ratio:.2f})")
            return is_speech
            
        except Exception as e:
            if self.test_mode:
                self._append_debug(f"VAD检测异常: {str(e)}")
            return False
    
    def process_audio_chunk(self, audio_chunk, callback_func=None):
        """
        处理音频块
        
        Args:
            audio_chunk: 音频数据块
            callback_func: 回调函数，用于处理语音片段
        """
        try:
            if len(audio_chunk) == 0:
                if self.test_mode:
                    self._append_debug("收到空音频帧，跳过处理")
                return
            
            # 添加调试信息
            if self.test_mode:
                self._append_debug(f"处理音频块: {len(audio_chunk)} 字节, 类型: {type(audio_chunk)}")
            
            # 管理音频缓存
            self._manage_audio_cache(audio_chunk)
            self.audio_buffer.append(audio_chunk)
            
            # VAD检测
            is_speech = self._is_speech(audio_chunk)
            current_time = time.time()
            
            if self.test_mode:
                self._append_debug(f"VAD检测结果: {'语音' if is_speech else '静音'}")
            
            if is_speech:
                self.consecutive_silence = 0
                self.speech_frames.append(audio_chunk)
                self.last_speech_time = current_time
                
                if not self.is_speaking:
                    self.is_speaking = True
                    if self.test_mode:
                        self._append_debug("检测到语音 - 开始录音")
                
                # 实时处理：更频繁地进行部分处理以减少延迟
                if len(self.speech_frames) >= PARTIAL_UPDATE_FRAMES:  # 400ms 就进行一次处理
                    if self.test_mode:
                        self._append_debug(f"达到部分处理阈值: {len(self.speech_frames)} 帧")
                    # 保留更长的上下文
                    audio_data = b"".join(self.speech_frames)
                    if self.test_mode:
                        self._append_debug(f"开始处理语音片段: {len(audio_data)} 字节")
                    
                    if callback_func:
                        callback_func(audio_data, partial=True)
                    
                    self.speech_frames = self.speech_frames[-10:]  # 保留200ms上文
            else:
                # 空音频/非人声处理
                self.consecutive_silence += 1
                
                if self.is_speaking:
                    silence_threshold = SILENCE_THRESHOLD_LONG if len(self.speech_frames) > 80 else SILENCE_THRESHOLD_SHORT
                    
                    if self.consecutive_silence >= silence_threshold:
                        if len(self.speech_frames) > MIN_SPEECH_FRAMES:
                            if self.test_mode:
                                self._append_debug(f"静音达到阈值，处理语音片段: {len(self.speech_frames)} 帧")
                            audio_data = b"".join(self.speech_frames)
                            if self.test_mode:
                                self._append_debug(f"开始处理完整语音片段: {len(audio_data)} 字节")
                            
                            if callback_func:
                                callback_func(audio_data, partial=False)
                            
                        self.speech_frames = []
                        self.is_speaking = False
                        if self.test_mode:
                            self._append_debug("停止录音")
                
                # 长时间静音，清理状态
                if current_time - self.last_speech_time > 2.0:  # 2 秒无语音
                    # 无语音先处理之前的
                    if len(self.speech_frames) > MIN_SPEECH_FRAMES:
                        if self.test_mode:
                            self._append_debug(f"长时间静音，处理剩余语音片段: {len(self.speech_frames)} 帧")
                        audio_data = b"".join(self.speech_frames)
                        if self.test_mode:
                            self._append_debug(f"开始处理剩余语音片段: {len(audio_data)} 字节")
                        
                        if callback_func:
                            callback_func(audio_data, partial=False)
                        
                        self.speech_frames = []
                    
                    # 再清空翻译流
                    self.speech_frames = []
                    self.is_speaking = False
                    if self.test_mode:
                        self._append_debug("长时间静音，清理状态")
                    
        except Exception as e:
            if self.test_mode:
                self._append_debug(f"音频帧处理异常: {str(e)}")
    
    def _manage_audio_cache(self, audio_data):
        """管理音频缓存"""
        self.audio_cache.append(audio_data)
        self.frame_counter += 1
        
        # 定期清理缓存
        if self.frame_counter % CACHE_CLEANUP_INTERVAL == 0:
            # 保留最近的帧，清理旧的
            while len(self.audio_cache) > MAX_CACHE_SIZE // 2:
                self.audio_cache.popleft()
            
            if self.test_mode:
                self._append_debug(f"音频缓存清理: 当前大小={len(self.audio_cache)}")
    
    def get_cached_audio(self, num_frames: int = 10) -> List:
        """获取缓存的音频数据"""
        if len(self.audio_cache) >= num_frames:
            return list(self.audio_cache)[-num_frames:]
        return list(self.audio_cache)
    
    def get_debug_info(self) -> str:
        """获取调试信息"""
        return '\n'.join(self.debug_msgs)
    
    def reset_state(self):
        """重置音频处理状态"""
        self.is_speaking = False
        self.silence_frames = 0
        self.speech_frames = []
        self.consecutive_silence = 0
        self.last_speech_time = time.time()
        self.audio_buffer.clear()
        self.audio_cache.clear()
        self.frame_counter = 0
        self.vad_window.clear()
        self.debug_msgs.clear()
    
    def close(self):
        """关闭音频处理器"""
        try:
            if hasattr(self, 'audio'):
                self.audio.terminate()
        except Exception as e:
            logger.error(f"关闭音频处理器异常: {e}") 