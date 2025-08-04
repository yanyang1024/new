"""
核心音频翻译器模块
整合音频处理和翻译功能
"""

import numpy as np
import threading
import time
import logging
from typing import Optional, Callable
import asyncio

from ..audio.audio_processor import AudioProcessor
from ..translation.translator import Translator
from ..config.audio_config import RATE

logger = logging.getLogger(__name__)

class AudioTranslator:
    """音频翻译器核心类"""
    
    def __init__(self, test_mode: bool = True):
        """
        初始化音频翻译器
        
        Args:
            test_mode: 是否为测试模式
        """
        self.test_mode = test_mode
        
        # 初始化组件
        self.audio_processor = AudioProcessor(test_mode)
        self.translator = Translator(test_mode)
        
        # 状态管理
        self.running = False
        self.input_mode = None  # 'mic' or 'gradio_mic'
        
        # Gradio相关状态
        self.gradio_audio_stream = None
        self.gradio_audio_buffer = []
        self.gradio_processing = False
        self.gradio_audio_lock = threading.Lock()
        
        # 服务器相关
        self.websocket_server = None
        self.udp_server = None
        
        # 设置音频处理回调
        self.audio_processor.process_audio_chunk = self._audio_chunk_callback
    
    def _audio_chunk_callback(self, audio_chunk, callback_func=None):
        """
        音频块处理回调
        
        Args:
            audio_chunk: 音频数据块
            callback_func: 回调函数
        """
        # 调用音频处理器的处理方法
        self.audio_processor.process_audio_chunk(
            audio_chunk, 
            callback_func=self._speech_segment_callback
        )
    
    def _speech_segment_callback(self, audio_data: bytes, partial: bool = False):
        """
        语音片段处理回调
        
        Args:
            audio_data: 音频数据
            partial: 是否为部分处理
        """
        # 调用翻译器处理语音片段
        self.translator.process_speech_segment(audio_data, partial)
    
    def set_input_mode(self, mode: str):
        """
        设置输入模式
        
        Args:
            mode: 输入模式 ('mic' 或 'gradio_mic')
        """
        self.input_mode = mode
        if self.test_mode:
            self._append_debug(f"设置输入模式: {mode}")
        
        # 如果是Gradio麦克风模式，重置相关状态
        if mode == 'gradio_mic':
            self.reset_gradio_state()
    
    def _setup_websocket_server(self, host: str = "0.0.0.0", port: int = 8765):
        """设置 WebSocket 服务器"""
        try:
            from websocket_server import AudioWebSocketServer
            
            self.websocket_server = AudioWebSocketServer(
                host=host,
                port=port,
                audio_callback=self._audio_chunk_callback
            )
            
            # 在新线程中启动 WebSocket 服务器
            def run_server():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(self.websocket_server.start())
                loop.run_forever()
                
            self.websocket_thread = threading.Thread(target=run_server, daemon=True)
            self.websocket_thread.start()
            
        except ImportError:
            logger.warning("WebSocket服务器模块未找到")
        except Exception as e:
            logger.error(f"设置WebSocket服务器失败: {e}")
    
    def _setup_udp_server(self, host: str = "0.0.0.0", port: int = 12345):
        """设置 UDP 服务器"""
        try:
            from udp_server import AudioUDPServer
            
            self.udp_server = AudioUDPServer(
                host=host,
                port=port,
                audio_callback=self._audio_chunk_callback
            )
            self.udp_server.start()
            
        except ImportError:
            logger.warning("UDP服务器模块未找到")
        except Exception as e:
            logger.error(f"设置UDP服务器失败: {e}")
    
    def run(self):
        """运行音频翻译器"""
        try:
            if self.input_mode == 'mic':
                # 麦克风模式等待 WebSocket 连接
                self._run_mic_mode()
            elif self.input_mode == 'gradio_mic':
                # Gradio麦克风模式 - 不需要额外处理，由Gradio界面直接调用
                self._run_gradio_mic_mode()
            else:
                raise ValueError(f"不支持的输入模式: {self.input_mode}")
                
        except Exception as e:
            logger.error(f"run 方法异常: {e}")
            raise
    
    def _run_mic_mode(self):
        """麦克风模式处理逻辑"""
        logger.info("[INFO] 等待客户端连接...")
        
        # 启动UDP服务器
        if not self.udp_server:
            self._setup_udp_server()
        
        # 等待音频数据，UDP服务器会自动调用回调函数
        while self.running:
            try:
                # 只需要等待，UDP服务器会自动调用audio_callback
                time.sleep(0.01)  # 短暂等待，减少CPU占用
                    
            except Exception as e:
                if self.test_mode:
                    self._append_debug(f"麦克风模式处理异常: {str(e)}")
                time.sleep(0.1)
    
    def _run_gradio_mic_mode(self):
        """Gradio麦克风模式处理逻辑"""
        logger.info("[INFO] Gradio麦克风模式已启动...")
        
        # Gradio模式下不需要额外的处理循环，音频处理由Gradio界面直接调用
        while self.running:
            try:
                # 检查Gradio处理状态
                status = self.get_gradio_status()
                if self.test_mode and status['processing']:
                    self._append_debug(f"Gradio处理状态: {status}")
                
                time.sleep(0.1)  # 短暂等待，减少CPU占用
                    
            except Exception as e:
                if self.test_mode:
                    self._append_debug(f"Gradio麦克风模式处理异常: {str(e)}")
                time.sleep(0.1)
    
    def process_gradio_audio_stream(self, stream, new_chunk):
        """
        处理Gradio Audio流式音频数据
        
        Args:
            stream: 音频流
            new_chunk: 新的音频块
            
        Returns:
            更新后的音频流和状态
        """
        try:
            if new_chunk is None:
                return stream, ""
            
            sr, y = new_chunk
            
            if self.test_mode:
                self._append_debug(f"Gradio音频流处理: sr={sr}, y_shape={y.shape}, y_dtype={y.dtype}")
            
            # 验证音频数据
            if not self._validate_audio_data(sr, y):
                return stream, ""
            
            # 转换为单声道
            if y.ndim > 1:
                y = y.mean(axis=1)
                if self.test_mode:
                    self._append_debug(f"转换为单声道: 新形状={y.shape}")
            
            # 改进的音频数据标准化处理
            # 根据Gradio文档，音频数据是16位int数组，范围从-32768到32767
            # 需要转换为float32并归一化到[-1, 1]范围，与文件模式保持一致
            y = y.astype(np.float32)
            
            if self.test_mode:
                self._append_debug(f"原始数据范围: [{np.min(y):.4f}, {np.max(y):.4f}]")
            
            # 使用与文件模式相同的归一化方式
            # 文件模式使用: audio_array.astype(np.float32) / 32768.0
            y = y / 32768.0
            
            if self.test_mode:
                self._append_debug(f"归一化后范围: [{np.min(y):.4f}, {np.max(y):.4f}]")
            
            # 检查归一化后的数据范围
            max_amplitude = np.max(np.abs(y))
            if max_amplitude > 1.0:
                if self.test_mode:
                    self._append_debug(f"归一化后数据范围异常: {max_amplitude:.4f}，进行裁剪")
                # 裁剪到[-1, 1]范围
                y = np.clip(y, -1.0, 1.0)
            
            if self.test_mode:
                self._append_debug(f"音频数据归一化完成: 最大值={np.max(np.abs(y)):.4f}")
            
            # 重采样到16kHz（如果需要）
            if sr != RATE:
                y = self._resample_audio(y, sr, RATE)
            
            # 累积音频流
            if stream is not None:
                stream = np.concatenate([stream, y])
            else:
                stream = y
            
            # 限制音频流长度（避免内存溢出）
            max_length = RATE * 30  # 最多30秒
            if len(stream) > max_length:
                # 保留最新的音频数据
                stream = stream[-max_length:]
                if self.test_mode:
                    self._append_debug(f"音频流长度限制: 保留最新{max_length}采样点")
            
            # 处理音频块
            self._process_gradio_audio_chunk(y)
            
            return stream, ""
            
        except Exception as e:
            if self.test_mode:
                self._append_debug(f"Gradio音频处理异常: {str(e)}")
            return stream, ""
    
    def _validate_audio_data(self, sr: int, y) -> bool:
        """验证音频数据的有效性"""
        try:
            if y is None or len(y) == 0:
                if self.test_mode:
                    self._append_debug("音频数据为空")
                return False
            
            if sr <= 0:
                if self.test_mode:
                    self._append_debug(f"采样率无效: {sr}")
                return False
            
            if np.isnan(y).any() or np.isinf(y).any():
                if self.test_mode:
                    self._append_debug("音频数据包含NaN或Inf值")
                return False
            
            # 放宽音频数据范围检查，麦克风数据可能超出[-1, 1]范围
            max_amplitude = np.max(np.abs(y))
            if max_amplitude > 10.0:  # 允许更大的范围，但记录警告
                if self.test_mode:
                    self._append_debug(f"音频数据超出正常范围: {max_amplitude:.2f}，将进行归一化处理")
            
            return True
        except Exception as e:
            if self.test_mode:
                self._append_debug(f"音频数据验证异常: {str(e)}")
            return False
    
    def _resample_audio(self, audio, original_sr: int, target_sr: int):
        """重采样音频数据"""
        try:
            if original_sr == target_sr:
                return audio
            
            # 检查输入音频数据是否有效
            if np.isnan(audio).any() or np.isinf(audio).any():
                if self.test_mode:
                    self._append_debug("重采样输入包含无效值，返回原音频")
                return audio
            
            # 简单的线性插值重采样
            ratio = target_sr / original_sr
            new_length = int(len(audio) * ratio)
            
            # 确保新长度不为0
            if new_length <= 0:
                if self.test_mode:
                    self._append_debug(f"重采样长度异常: {new_length}")
                return audio
            
            # 使用更安全的插值方法
            try:
                indices = np.linspace(0, len(audio) - 1, new_length)
                resampled = np.interp(indices, np.arange(len(audio)), audio)
                
                # 检查重采样结果是否有效
                if np.isnan(resampled).any() or np.isinf(resampled).any():
                    if self.test_mode:
                        self._append_debug("重采样结果包含无效值，返回原音频")
                    return audio
                
                if self.test_mode:
                    self._append_debug(f"重采样: {original_sr}Hz -> {target_sr}Hz, {len(audio)} -> {len(resampled)}")
                
                return resampled
                
            except Exception as e:
                if self.test_mode:
                    self._append_debug(f"重采样插值异常: {str(e)}")
                return audio
                
        except Exception as e:
            if self.test_mode:
                self._append_debug(f"重采样异常: {str(e)}")
            return audio
    
    def _process_gradio_audio_chunk(self, audio_chunk):
        """处理Gradio音频块"""
        try:
            with self.gradio_audio_lock:
                # 设置处理状态
                self.gradio_processing = True
                
                # 添加到Gradio音频缓冲区
                self.gradio_audio_buffer.append(audio_chunk)
                
                if self.test_mode:
                    self._append_debug(f"Gradio音频块处理开始: 长度={len(audio_chunk)}, 类型={type(audio_chunk)}")
                
                # 改进的音频数据转换逻辑
                # Gradio音频已经是float32格式且已归一化到[-1, 1]，需要转换为int16字节格式
                try:
                    # 确保音频数据是numpy数组
                    if not isinstance(audio_chunk, np.ndarray):
                        if self.test_mode:
                            self._append_debug(f"音频数据类型错误: {type(audio_chunk)}，尝试转换")
                        audio_chunk = np.array(audio_chunk, dtype=np.float32)
                    
                    # 检查音频数据是否有效并尝试修复
                    if np.isnan(audio_chunk).any():
                        if self.test_mode:
                            self._append_debug(f"音频数据包含NaN值，尝试修复。数据范围: [{np.min(audio_chunk):.4f}, {np.max(audio_chunk):.4f}]")
                        # 尝试修复NaN值
                        audio_chunk = np.nan_to_num(audio_chunk, nan=0.0, posinf=1.0, neginf=-1.0)
                    
                    if np.isinf(audio_chunk).any():
                        if self.test_mode:
                            self._append_debug(f"音频数据包含Inf值，尝试修复。数据范围: [{np.min(audio_chunk):.4f}, {np.max(audio_chunk):.4f}]")
                        # 尝试修复Inf值
                        audio_chunk = np.nan_to_num(audio_chunk, nan=0.0, posinf=1.0, neginf=-1.0)
                    
                    # 最终检查修复后的数据
                    if np.isnan(audio_chunk).any() or np.isinf(audio_chunk).any():
                        if self.test_mode:
                            self._append_debug("音频数据修复后仍包含无效值，跳过处理")
                        return
                    
                    # 确保音频数据在合理范围内
                    audio_chunk = np.clip(audio_chunk, -1.0, 1.0)
                    
                    if self.test_mode:
                        self._append_debug(f"音频数据预处理完成: 范围=[{np.min(audio_chunk):.4f}, {np.max(audio_chunk):.4f}]")
                    
                    # 转换为int16格式，与文件模式保持一致
                    # 文件模式使用: data.tobytes()，其中data是float32格式
                    # 我们需要将float32转换为int16，然后转换为bytes
                    audio_int16 = (audio_chunk * 32768.0).astype(np.int16)
                    
                    # 检查转换是否成功
                    if np.isnan(audio_int16).any() or np.isinf(audio_int16).any():
                        if self.test_mode:
                            self._append_debug("int16转换失败，包含无效值")
                        return
                    
                    audio_bytes = audio_int16.tobytes()
                    
                    if self.test_mode:
                        self._append_debug(f"Gradio音频转换成功: 长度={len(audio_chunk)}, 最大值={np.max(np.abs(audio_chunk)):.4f}")
                    
                    # 使用现有的音频处理逻辑
                    self._audio_chunk_callback(audio_bytes)
                    
                except Exception as e:
                    if self.test_mode:
                        self._append_debug(f"Gradio音频转换异常: {str(e)}")
                    # 如果转换失败，尝试直接处理float32数据
                    try:
                        # 直接使用float32数据，但需要确保格式正确
                        if isinstance(audio_chunk, np.ndarray):
                            # 将float32数据转换为bytes格式
                            audio_bytes = audio_chunk.tobytes()
                            if self.test_mode:
                                self._append_debug("使用备用方法处理float32数据")
                            self._audio_chunk_callback(audio_bytes)
                        else:
                            if self.test_mode:
                                self._append_debug("音频数据类型不支持")
                    except Exception as e2:
                        if self.test_mode:
                            self._append_debug(f"备用音频处理也失败: {str(e2)}")
                
        except Exception as e:
            if self.test_mode:
                self._append_debug(f"Gradio音频块处理异常: {str(e)}")
        finally:
            with self.gradio_audio_lock:
                self.gradio_processing = False
    
    def reset_gradio_state(self):
        """重置Gradio相关状态"""
        with self.gradio_audio_lock:
            self.gradio_audio_buffer.clear()
            self.gradio_processing = False
            self.gradio_audio_stream = None
    
    def get_gradio_status(self):
        """获取Gradio处理状态"""
        with self.gradio_audio_lock:
            return {
                'processing': self.gradio_processing,
                'buffer_size': len(self.gradio_audio_buffer),
                'stream_active': self.gradio_audio_stream is not None
            }
    
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
        """获取所有翻译记录"""
        return self.translator.get_all_translations()
    
    def close(self):
        """关闭音频翻译器"""
        logger.info("[INFO] 开始关闭AudioTranslator...")
        self.running = False
        
        # 关闭 UDP 服务器
        if self.udp_server:
            logger.info("[INFO] 关闭UDP服务器...")
            self.udp_server.stop()
            self.udp_server = None
            
        # 关闭 WebSocket 服务器
        if self.websocket_server:
            logger.info("[INFO] 关闭WebSocket服务器...")
            try:
                asyncio.run(self.websocket_server.stop())
            except Exception as e:
                logger.warning(f"关闭WebSocket服务器异常: {e}")
            self.websocket_server = None
            
        # 重置Gradio状态
        if self.input_mode == 'gradio_mic':
            logger.info("[INFO] 重置Gradio状态...")
            self.reset_gradio_state()
            
        try:
            # 关闭音频处理器
            self.audio_processor.close()
            
            # 关闭翻译器
            self.translator.close()
                
            if self.test_mode:
                logger.info("\nDebug Information:")
                for key, value in self.audio_processor.debug_info.items():
                    logger.info(f"{key}: {value}")
                
        except Exception as e:
            logger.error(f"关闭异常: {e}")
        
        logger.info("[INFO] AudioTranslator关闭完成") 