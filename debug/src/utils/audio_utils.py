import numpy as np
import torch
import soundfile as sf
import os
from datetime import datetime
from ..config.audio_config import *

def validate_audio_data(audio_data, test_mode=False):
    """验证音频数据的有效性"""
    try:
        if audio_data is None or len(audio_data) == 0:
            if test_mode:
                print("[DEBUG] 音频数据为空")
            return False
        
        if np.isnan(audio_data).any() or np.isinf(audio_data).any():
            if test_mode:
                print("[DEBUG] 音频数据包含NaN或Inf值")
            return False
        
        # 放宽音频数据范围检查，麦克风数据可能超出[-1, 1]范围
        max_amplitude = np.max(np.abs(audio_data))
        if max_amplitude > 10.0:  # 允许更大的范围，但记录警告
            if test_mode:
                print(f"[DEBUG] 音频数据超出正常范围: {max_amplitude:.2f}，将进行归一化处理")
        
        return True
    except Exception as e:
        if test_mode:
            print(f"[DEBUG] 音频数据验证异常: {str(e)}")
        return False

def normalize_audio_data(audio_data, test_mode=False):
    """统一音频数据格式处理，确保与文件模式一致"""
    try:
        if audio_data is None or len(audio_data) == 0:
            return None
        
        # 确保数据类型为float32
        if not isinstance(audio_data, np.ndarray):
            audio_data = np.array(audio_data, dtype=np.float32)
        else:
            audio_data = audio_data.astype(np.float32)
        
        # 处理音频数据范围
        max_amplitude = np.max(np.abs(audio_data))
        
        if max_amplitude > 0:
            # 如果数据范围过大，进行软裁剪
            if max_amplitude > MAX_AMPLITUDE:
                audio_data = np.clip(audio_data, -MAX_AMPLITUDE, MAX_AMPLITUDE)
                max_amplitude = MAX_AMPLITUDE
                if test_mode:
                    print(f"[DEBUG] 音频数据软裁剪: 最大值={max_amplitude:.2f}")
            
            # 归一化到[-1, 1]范围
            audio_data = audio_data / max_amplitude
        else:
            # 静音帧，保持原样
            if test_mode:
                print("[DEBUG] 检测到静音帧")
        
        return audio_data
        
    except Exception as e:
        if test_mode:
            print(f"[DEBUG] 音频数据归一化异常: {str(e)}")
        return None

def resample_audio(audio, original_sr, target_sr, test_mode=False):
    """重采样音频数据"""
    try:
        if original_sr == target_sr:
            return audio
        
        # 检查输入音频数据是否有效
        if np.isnan(audio).any() or np.isinf(audio).any():
            if test_mode:
                print("[DEBUG] 重采样输入包含无效值，返回原音频")
            return audio
        
        # 简单的线性插值重采样
        ratio = target_sr / original_sr
        new_length = int(len(audio) * ratio)
        
        # 确保新长度不为0
        if new_length <= 0:
            if test_mode:
                print(f"[DEBUG] 重采样长度异常: {new_length}")
            return audio
        
        # 使用更安全的插值方法
        try:
            indices = np.linspace(0, len(audio) - 1, new_length)
            resampled = np.interp(indices, np.arange(len(audio)), audio)
            
            # 检查重采样结果是否有效
            if np.isnan(resampled).any() or np.isinf(resampled).any():
                if test_mode:
                    print("[DEBUG] 重采样结果包含无效值，返回原音频")
                return audio
            
            if test_mode:
                print(f"[DEBUG] 重采样: {original_sr}Hz -> {target_sr}Hz, {len(audio)} -> {len(resampled)}")
            
            return resampled
            
        except Exception as e:
            if test_mode:
                print(f"[DEBUG] 重采样插值异常: {str(e)}")
            return audio
            
    except Exception as e:
        if test_mode:
            print(f"[DEBUG] 重采样异常: {str(e)}")
        return audio

def save_audio_segment(audio_data, filename, test_mode=False):
    """保存音频片段到文件"""
    try:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        sf.write(filename, audio_data, RATE)
        if test_mode:
            print(f"[DEBUG] 保存音频片段: {filename}")
        return True
    except Exception as e:
        if test_mode:
            print(f"[DEBUG] 保存音频片段失败: {str(e)}")
        return False

def convert_audio_format(audio_data, target_format='int16'):
    """转换音频数据格式"""
    try:
        if target_format == 'int16':
            # 转换为int16格式
            audio_int16 = (audio_data * NORMALIZATION_FACTOR).astype(np.int16)
            return audio_int16.tobytes()
        elif target_format == 'float32':
            # 转换为float32格式
            if isinstance(audio_data, bytes):
                return np.frombuffer(audio_data, dtype=np.float32)
            else:
                return audio_data.astype(np.float32)
        else:
            raise ValueError(f"不支持的音频格式: {target_format}")
    except Exception as e:
        print(f"[ERROR] 音频格式转换失败: {str(e)}")
        return None

def calculate_audio_energy(audio_data, test_mode=False):
    """计算音频能量"""
    try:
        # 使用更安全的能量计算方法
        abs_audio = np.abs(audio_data)
        # 检查是否有异常大的值
        if np.max(abs_audio) > 1e6:  # 如果最大值超过1e6，说明有异常
            if test_mode:
                print(f"[DEBUG] 音频数据异常大: {np.max(abs_audio):.2e}，进行裁剪")
            abs_audio = np.clip(abs_audio, 0, 1e6)
        
        # 使用更稳定的平均值计算
        energy = np.mean(abs_audio)
        
        # 检查能量值是否合理
        if np.isnan(energy) or np.isinf(energy):
            if test_mode:
                print(f"[DEBUG] 能量计算异常: {energy}，使用默认值")
            energy = 0.0
        
        return energy
        
    except Exception as e:
        if test_mode:
            print(f"[DEBUG] 能量计算异常: {str(e)}，使用默认值")
        return 0.0 