import numpy as np
import pyaudio
import wave
import threading
import time
from pydub import AudioSegment
from collections import deque
from ..config.audio_config import *
from ..utils.audio_utils import *

class AudioProcessor:
    def __init__(self, test_mode=True):
        self.test_mode = test_mode
        self.audio = pyaudio.PyAudio()
        
    def _get_audio_input_stream(self, input_mode, file_path=None):
        """根据输入模式返回相应的音频流"""
        if input_mode == 'file':
            return self._get_file_stream(file_path)
        elif input_mode == 'mic':
            # 麦克风模式 - 使用 UDP
            return None  # UDP 模式下不需要返回流
        elif input_mode == 'gradio_mic':
            # Gradio麦克风模式 - 使用Gradio Audio流式处理
            return None  # Gradio模式下不需要返回流，直接处理
        else:
            raise ValueError(f"不支持的输入模式: {input_mode}")
    
    def _get_file_stream(self, file_path):
        """获取文件音频流"""
        if self.test_mode:
            class FileStream:
                def __init__(self, wav_path, chunk):
                    try:
                        # 使用pydub加载音频文件并转换为16位PCM编码
                        audio = AudioSegment.from_file(wav_path)
                        audio = audio.set_channels(1).set_frame_rate(16000).set_sample_width(2)
                        audio_raw_data = audio.raw_data
                        
                        self.framerate = audio.frame_rate
                        self.nchannels = audio.channels
                        self.sampwidth = audio.sample_width
                        
                        # 转换为numpy数组
                        audio_array = np.frombuffer(audio_raw_data, dtype=np.int16)
                        
                        # 转换为float32并归一化
                        self.audio_data = audio_array.astype(np.float32) / 32768.0
                        
                        self.chunk = chunk
                        self.position = 0
                        self.lock = threading.Lock()
                        print(f"[DEBUG] 成功加载音频文件: {wav_path}")
                        print(f"[DEBUG] 音频长度: {len(self.audio_data)} 采样点")
                        print(f"[DEBUG] 采样率: {self.framerate} Hz")
                        print(f"[DEBUG] 声道数: {self.nchannels}")
                    except Exception as e:
                        print(f"[ERROR] 处理音频文件 {wav_path} 失败: {e}")
                        raise

                def read(self, chunk, exception_on_overflow=False):
                    with self.lock:
                        if self.position >= len(self.audio_data):
                            # 测试模式下，读到文件末尾就停止，不重复播放
                            return b''
                        
                        end_pos = min(self.position + chunk, len(self.audio_data))
                        data = self.audio_data[self.position:end_pos]
                        self.position += chunk
                        
                        # 转换为bytes
                        return data.tobytes()

                def stop_stream(self): 
                    self.position = len(self.audio_data)  # 强制结束读取

                def close(self):
                    self.position = len(self.audio_data)
                    print("[DEBUG] 关闭音频文件流")

            return FileStream(file_path, CHUNK)
        else:
            try:
                return self.audio.open(
                    format=getattr(pyaudio, FORMAT),
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK
                )
            except Exception as e:
                print(f"[ERROR] 打开麦克风音频流失败: {e}")
                raise

    def process_audio_chunk(self, audio_chunk, translator):
        """处理音频块的完整逻辑"""
        try:
            if len(audio_chunk) == 0:
                if self.test_mode:
                    translator._append_debug("收到空音频帧，跳过处理")
                return
            
            # 添加调试信息，特别是在麦克风模式下
            if self.test_mode and (translator.input_mode == 'mic' or translator.input_mode == 'gradio_mic'):
                translator._append_debug(f"处理音频块: {len(audio_chunk)} 字节, 类型: {type(audio_chunk)}, 模式: {translator.input_mode}")
            
            # 管理音频缓存
            self._manage_audio_cache(audio_chunk, translator)
            translator.audio_buffer.append(audio_chunk)
            
            # VAD检测
            is_speech = translator._is_speech(audio_chunk)
            current_time = time.time()
            
            if self.test_mode:
                translator._append_debug(f"VAD检测结果: {'语音' if is_speech else '静音'}")
            
            if is_speech:
                translator.consecutive_silence = 0
                translator.speech_frames.append(audio_chunk)
                translator.last_speech_time = current_time
                
                if not translator.is_speaking:
                    translator.is_speaking = True # 设置标志位，表示正在说话 （也会传递到下一处理状态）
                    if self.test_mode:
                        translator._append_debug("检测到语音 - 开始录音")
                
                # 实时处理：更频繁地进行部分处理以减少延迟
                if len(translator.speech_frames) >= PARTIAL_UPDATE_FRAMES:  # 400ms 就进行一次处理
                    if self.test_mode:
                        translator._append_debug(f"达到部分处理阈值: {len(translator.speech_frames)} 帧")
                    # 保留更长的上下文
                    audio_data = b"".join(translator.speech_frames)
                    if self.test_mode:
                        translator._append_debug(f"开始处理语音片段: {len(audio_data)} 字节")
                    translator.process_speech_segment(audio_data, partial=True) # partial=True 表明 进行部分处理

                    translator.speech_frames = translator.speech_frames[-10:]  # 保留200ms上文
            else:
                # 空音频/非人声处理
                translator.consecutive_silence += 1
                
                if translator.is_speaking:
                    silence_threshold = SILENCE_THRESHOLD_LONG if len(translator.speech_frames) > 80 else SILENCE_THRESHOLD_SHORT
                    
                    if translator.consecutive_silence >= silence_threshold:
                        if len(translator.speech_frames) > MIN_SPEECH_FRAMES:
                            if self.test_mode:
                                translator._append_debug(f"静音达到阈值，处理语音片段: {len(translator.speech_frames)} 帧")
                            audio_data = b"".join(translator.speech_frames)
                            if self.test_mode:
                                translator._append_debug(f"开始处理完整语音片段: {len(audio_data)} 字节")
                            translator.process_speech_segment(audio_data, partial=False)
                            
                        translator.speech_frames = []
                        translator.is_speaking = False
                        if self.test_mode:
                            translator._append_debug("停止录音")
                
                # 长时间静音，清理状态
                if current_time - translator.last_speech_time > 2.0:  # 2 秒无语音
                    # 无语音先处理之前的
                    if len(translator.speech_frames) > MIN_SPEECH_FRAMES:
                        if self.test_mode:
                            translator._append_debug(f"长时间静音，处理剩余语音片段: {len(translator.speech_frames)} 帧")
                        audio_data = b"".join(translator.speech_frames)
                        if self.test_mode:
                            translator._append_debug(f"开始处理剩余语音片段: {len(audio_data)} 字节")
                        translator.process_speech_segment(audio_data, partial=False)
                        translator.speech_frames = []
                

                    # 再清空翻译流
                    translator.speech_frames = []
                    translator.is_speaking = False
                    if self.test_mode:
                        translator._append_debug("长时间静音，清理状态")
                    
        except Exception as e:
            if self.test_mode:
                translator._append_debug(f"音频帧处理异常: {str(e)}")

    def _manage_audio_cache(self, audio_data, translator):
        """管理音频缓存"""
        with translator.cache_lock:
            translator.audio_cache.append(audio_data)
            translator.frame_counter += 1
            
            # 定期清理缓存
            if translator.frame_counter % translator.cache_cleanup_interval == 0:
                # 保留最近的帧，清理旧的
                while len(translator.audio_cache) > translator.max_cache_size // 2:
                    translator.audio_cache.popleft()
                
                if self.test_mode:
                    translator._append_debug(f"音频缓存清理: 当前大小={len(translator.audio_cache)}")

    def process_gradio_audio_stream(self, stream, new_chunk, translator):
        """处理Gradio Audio流式音频数据"""
        try:
            if new_chunk is None:
                return stream, ""
            
            sr, y = new_chunk
            
            if self.test_mode:
                translator._append_debug(f"Gradio音频流处理: sr={sr}, y_shape={y.shape}, y_dtype={y.dtype}")
            
            # 验证音频数据
            if not validate_audio_data(sr, y, self.test_mode):
                return stream, ""
            
            # 转换为单声道
            if y.ndim > 1:
                y = y.mean(axis=1)
                if self.test_mode:
                    translator._append_debug(f"转换为单声道: 新形状={y.shape}")
            
            # 改进的音频数据标准化处理
            # 根据Gradio文档，音频数据是16位int数组，范围从-32768到32767
            # 需要转换为float32并归一化到[-1, 1]范围，与文件模式保持一致
            y = y.astype(np.float32)
            
            if self.test_mode:
                translator._append_debug(f"原始数据范围: [{np.min(y):.4f}, {np.max(y):.4f}]")
            
            # 使用与文件模式相同的归一化方式
            # 文件模式使用: audio_array.astype(np.float32) / 32768.0
            y = y / NORMALIZATION_FACTOR
            
            if self.test_mode:
                translator._append_debug(f"归一化后范围: [{np.min(y):.4f}, {np.max(y):.4f}]")
            
            # 检查归一化后的数据范围
            max_amplitude = np.max(np.abs(y))
            if max_amplitude > 1.0:
                if self.test_mode:
                    translator._append_debug(f"归一化后数据范围异常: {max_amplitude:.4f}，进行裁剪")
                # 裁剪到[-1, 1]范围
                y = np.clip(y, -1.0, 1.0)
            
            if self.test_mode:
                translator._append_debug(f"音频数据归一化完成: 最大值={np.max(np.abs(y)):.4f}")
            
            # 重采样到16kHz（如果需要）
            if sr != RATE:
                y = resample_audio(y, sr, RATE, self.test_mode)
            
            # 累积音频流
            if stream is not None:
                stream = np.concatenate([stream, y])
            else:
                stream = y
            
            # 限制音频流长度（避免内存溢出）
            max_length = RATE * MAX_AUDIO_LENGTH  # 最多30秒
            if len(stream) > max_length:
                # 保留最新的音频数据
                stream = stream[-max_length:]
                if self.test_mode:
                    translator._append_debug(f"音频流长度限制: 保留最新{max_length}采样点")
            
            # 处理音频块
            self._process_gradio_audio_chunk(y, translator)
            
            return stream, ""
            
        except Exception as e:
            if self.test_mode:
                translator._append_debug(f"Gradio音频处理异常: {str(e)}")
            return stream, ""

    def _process_gradio_audio_chunk(self, audio_chunk, translator):
        """处理Gradio音频块"""
        try:
            with translator.gradio_audio_lock:
                # 设置处理状态
                translator.gradio_processing = True
                
                # 添加到Gradio音频缓冲区
                translator.gradio_audio_buffer.append(audio_chunk)
                
                if self.test_mode:
                    translator._append_debug(f"Gradio音频块处理开始: 长度={len(audio_chunk)}, 类型={type(audio_chunk)}")
                
                # 改进的音频数据转换逻辑
                # Gradio音频已经是float32格式且已归一化到[-1, 1]，需要转换为int16字节格式
                try:
                    # 确保音频数据是numpy数组
                    if not isinstance(audio_chunk, np.ndarray):
                        if self.test_mode:
                            translator._append_debug(f"音频数据类型错误: {type(audio_chunk)}，尝试转换")
                        audio_chunk = np.array(audio_chunk, dtype=np.float32)
                    
                    # 检查音频数据是否有效并尝试修复
                    if np.isnan(audio_chunk).any():
                        if self.test_mode:
                            translator._append_debug(f"音频数据包含NaN值，尝试修复。数据范围: [{np.min(audio_chunk):.4f}, {np.max(audio_chunk):.4f}]")
                        # 尝试修复NaN值
                        audio_chunk = np.nan_to_num(audio_chunk, nan=0.0, posinf=1.0, neginf=-1.0)
                    
                    if np.isinf(audio_chunk).any():
                        if self.test_mode:
                            translator._append_debug(f"音频数据包含Inf值，尝试修复。数据范围: [{np.min(audio_chunk):.4f}, {np.max(audio_chunk):.4f}]")
                        # 尝试修复Inf值
                        audio_chunk = np.nan_to_num(audio_chunk, nan=0.0, posinf=1.0, neginf=-1.0)
                    
                    # 最终检查修复后的数据
                    if np.isnan(audio_chunk).any() or np.isinf(audio_chunk).any():
                        if self.test_mode:
                            translator._append_debug("音频数据修复后仍包含无效值，跳过处理")
                        return
                    
                    # 确保音频数据在合理范围内
                    audio_chunk = np.clip(audio_chunk, -1.0, 1.0)
                    
                    if self.test_mode:
                        translator._append_debug(f"音频数据预处理完成: 范围=[{np.min(audio_chunk):.4f}, {np.max(audio_chunk):.4f}]")
                    
                    # 转换为int16格式，与文件模式保持一致
                    # 文件模式使用: data.tobytes()，其中data是float32格式
                    # 我们需要将float32转换为int16，然后转换为bytes
                    audio_int16 = (audio_chunk * NORMALIZATION_FACTOR).astype(np.int16)
                    
                    # 检查转换是否成功
                    if np.isnan(audio_int16).any() or np.isinf(audio_int16).any():
                        if self.test_mode:
                            translator._append_debug("int16转换失败，包含无效值")
                        return
                    
                    audio_bytes = audio_int16.tobytes()
                    
                    if self.test_mode:
                        translator._append_debug(f"Gradio音频转换成功: 长度={len(audio_chunk)}, 最大值={np.max(np.abs(audio_chunk)):.4f}")
                    
                    # 使用现有的音频处理逻辑
                    self.process_audio_chunk(audio_bytes, translator)
                    
                except Exception as e:
                    if self.test_mode:
                        translator._append_debug(f"Gradio音频转换异常: {str(e)}")
                    # 如果转换失败，尝试直接处理float32数据
                    try:
                        # 直接使用float32数据，但需要确保格式正确
                        if isinstance(audio_chunk, np.ndarray):
                            # 将float32数据转换为bytes格式
                            audio_bytes = audio_chunk.tobytes()
                            if self.test_mode:
                                translator._append_debug("使用备用方法处理float32数据")
                            self.process_audio_chunk(audio_bytes, translator)
                        else:
                            if self.test_mode:
                                translator._append_debug("音频数据类型不支持")
                    except Exception as e2:
                        if self.test_mode:
                            translator._append_debug(f"备用音频处理也失败: {str(e2)}")
                
        except Exception as e:
            if self.test_mode:
                translator._append_debug(f"Gradio音频块处理异常: {str(e)}")
        finally:
            with translator.gradio_audio_lock:
                translator.gradio_processing = False

    def reset_gradio_state(self, translator):
        """重置Gradio相关状态"""
        with translator.gradio_audio_lock:
            translator.gradio_audio_buffer.clear()
            translator.gradio_processing = False
            translator.gradio_audio_stream = None

    def get_gradio_status(self, translator):
        """获取Gradio处理状态"""
        with translator.gradio_audio_lock:
            return {
                'processing': translator.gradio_processing,
                'buffer_size': len(translator.gradio_audio_buffer),
                'stream_active': translator.gradio_audio_stream is not None
            } 