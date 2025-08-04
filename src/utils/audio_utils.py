"""
音频工具模块
包含音频处理相关的工具函数
"""

import numpy as np
import soundfile as sf
import os
from datetime import datetime
from typing import Optional, Tuple, Union
import logging

from ..config.audio_config import RATE, NORMALIZATION_FACTOR, MAX_AMPLITUDE

logger = logging.getLogger(__name__)

def normalize_audio_data(audio_data: np.ndarray, test_mode: bool = False) -> Optional[np.ndarray]:
    """
    统一音频数据格式处理，确保与文件模式一致
    
    Args:
        audio_data: 输入音频数据
        test_mode: 是否为测试模式
        
    Returns:
        标准化后的音频数据，如果处理失败返回None
    """
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
                    logger.debug(f"音频数据软裁剪: 最大值={max_amplitude:.2f}")
            
            # 归一化到[-1, 1]范围
            audio_data = audio_data / max_amplitude
        else:
            # 静音帧，保持原样
            if test_mode:
                logger.debug("检测到静音帧")
        
        return audio_data
        
    except Exception as e:
        if test_mode:
            logger.error(f"音频数据归一化异常: {str(e)}")
        return None

def validate_audio_data(sr: int, y: np.ndarray, test_mode: bool = False) -> bool:
    """
    验证音频数据的有效性
    
    Args:
        sr: 采样率
        y: 音频数据
        test_mode: 是否为测试模式
        
    Returns:
        音频数据是否有效
    """
    try:
        if y is None or len(y) == 0:
            if test_mode:
                logger.debug("音频数据为空")
            return False
        
        if sr <= 0:
            if test_mode:
                logger.debug(f"采样率无效: {sr}")
            return False
        
        if np.isnan(y).any() or np.isinf(y).any():
            if test_mode:
                logger.debug("音频数据包含NaN或Inf值")
            return False
        
        # 放宽音频数据范围检查，麦克风数据可能超出[-1, 1]范围
        max_amplitude = np.max(np.abs(y))
        if max_amplitude > 10.0:  # 允许更大的范围，但记录警告
            if test_mode:
                logger.debug(f"音频数据超出正常范围: {max_amplitude:.2f}，将进行归一化处理")
        
        return True
    except Exception as e:
        if test_mode:
            logger.error(f"音频数据验证异常: {str(e)}")
        return False

def resample_audio(audio: np.ndarray, original_sr: int, target_sr: int, test_mode: bool = False) -> np.ndarray:
    """
    重采样音频数据
    
    Args:
        audio: 输入音频数据
        original_sr: 原始采样率
        target_sr: 目标采样率
        test_mode: 是否为测试模式
        
    Returns:
        重采样后的音频数据
    """
    try:
        if original_sr == target_sr:
            return audio
        
        # 检查输入音频数据是否有效
        if np.isnan(audio).any() or np.isinf(audio).any():
            if test_mode:
                logger.debug("重采样输入包含无效值，返回原音频")
            return audio
        
        # 简单的线性插值重采样
        ratio = target_sr / original_sr
        new_length = int(len(audio) * ratio)
        
        # 确保新长度不为0
        if new_length <= 0:
            if test_mode:
                logger.debug(f"重采样长度异常: {new_length}")
            return audio
        
        # 使用更安全的插值方法
        try:
            indices = np.linspace(0, len(audio) - 1, new_length)
            resampled = np.interp(indices, np.arange(len(audio)), audio)
            
            # 检查重采样结果是否有效
            if np.isnan(resampled).any() or np.isinf(resampled).any():
                if test_mode:
                    logger.debug("重采样结果包含无效值，返回原音频")
                return audio
            
            if test_mode:
                logger.debug(f"重采样: {original_sr}Hz -> {target_sr}Hz, {len(audio)} -> {len(resampled)}")
            
            return resampled
            
        except Exception as e:
            if test_mode:
                logger.error(f"重采样插值异常: {str(e)}")
            return audio
            
    except Exception as e:
        if test_mode:
            logger.error(f"重采样异常: {str(e)}")
        return audio

def save_test_audio(audio_data: np.ndarray, test_mode: bool = False) -> Optional[str]:
    """
    保存测试音频文件
    
    Args:
        audio_data: 音频数据
        test_mode: 是否为测试模式
        
    Returns:
        保存的文件路径，如果失败返回None
    """
    if not test_mode:
        return None
        
    try:
        wav_path = f"test_segments/segment_{int(datetime.now().timestamp())}.wav"
        os.makedirs(os.path.dirname(wav_path), exist_ok=True)
        sf.write(wav_path, audio_data, RATE)
        logger.debug(f"保存测试音频片段: {wav_path}")
        return wav_path
    except Exception as e:
        logger.error(f"保存测试音频失败: {str(e)}")
        return None

def save_test_translation(translated_text: str, test_mode: bool = False) -> Optional[str]:
    """
    保存测试翻译文本
    
    Args:
        translated_text: 翻译文本
        test_mode: 是否为测试模式
        
    Returns:
        保存的文件路径，如果失败返回None
    """
    if not test_mode:
        return None
        
    try:
        translated_text_path = f"test_segments_translated/segment_{int(datetime.now().timestamp())}.txt"
        os.makedirs(os.path.dirname(translated_text_path), exist_ok=True)
        with open(translated_text_path, 'w', encoding='utf-8') as f:
            f.write(translated_text)
        logger.debug(f"保存测试翻译文本: {translated_text_path}")
        return translated_text_path
    except Exception as e:
        logger.error(f"保存测试翻译文本失败: {str(e)}")
        return None

def convert_audio_to_bytes(audio_data: np.ndarray, test_mode: bool = False) -> Optional[bytes]:
    """
    将音频数据转换为字节格式
    
    Args:
        audio_data: 音频数据
        test_mode: 是否为测试模式
        
    Returns:
        字节格式的音频数据
    """
    try:
        # 确保音频数据在合理范围内
        audio_data = np.clip(audio_data, -1.0, 1.0)
        
        # 转换为int16格式
        audio_int16 = (audio_data * NORMALIZATION_FACTOR).astype(np.int16)
        
        # 检查转换是否成功
        if np.isnan(audio_int16).any() or np.isinf(audio_int16).any():
            if test_mode:
                logger.debug("int16转换失败，包含无效值")
            return None
        
        audio_bytes = audio_int16.tobytes()
        
        if test_mode:
            logger.debug(f"音频转换成功: 长度={len(audio_data)}, 最大值={np.max(np.abs(audio_data)):.4f}")
        
        return audio_bytes
        
    except Exception as e:
        if test_mode:
            logger.error(f"音频转换异常: {str(e)}")
        return None 