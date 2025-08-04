"""
翻译工具模块
包含翻译处理相关的工具函数
"""

import json
import os
from datetime import datetime
from typing import List, Optional, Tuple
import logging

from ..config.translation_config import (
    PUNCTUATION_THRESHOLD, MAX_TRANSLATION_HISTORY, 
    MAX_PREVIOUS_TRANSLATIONS, MIN_SENTENCE_LENGTH
)

logger = logging.getLogger(__name__)

def format_translation(text: str, is_partial: bool = False) -> str:
    """
    格式化翻译文本，添加标点符号
    
    Args:
        text: 翻译文本
        is_partial: 是否为部分翻译
        
    Returns:
        格式化后的翻译文本
    """
    # 移除已有的结束标点
    text = text.rstrip('.!?')
    
    # 对部分翻译结果的处理
    if is_partial:
        # 如果是不完整的句子，添加省略号
        return text + "..."
        
    # 对完整翻译结果的处理
    words = text.split()
    if len(words) < 3:  # 短句
        return text + "."
        
    # 根据关键词添加适当的标点
    question_words = {'what', 'how', 'why', 'when', 'where', 'who', 'which'}
    if words[0].lower() in question_words:
        return text + "?"
    elif any(word.lower() in {'can', 'could', 'would', 'will', 'shall'} for word in words[:2]):
        return text + "?"
    else:
        return text + "."

def should_merge_translations(new_text: str, prev_text: str, last_sentence_end: float) -> bool:
    """
    增强的翻译合并判断逻辑
    
    Args:
        new_text: 新的翻译文本
        prev_text: 前一个翻译文本
        last_sentence_end: 上一句结束时间
        
    Returns:
        是否应该合并翻译
    """
    if not prev_text or not new_text:
        return False
        
    # 检查时间间隔
    current_time = datetime.now().timestamp()
    if current_time - last_sentence_end > PUNCTUATION_THRESHOLD:
        return False
        
    # 去除标点后再判断
    prev_clean = prev_text.rstrip('.!?').strip()
    new_clean = new_text.strip()
    
    # 检查句子完整性
    if len(prev_clean.split()) < MIN_SENTENCE_LENGTH:  # 前一句过短，可能不完整
        return True
        
    # 计算词重叠
    prev_words = set(prev_clean.lower().split())
    new_words = set(new_clean.lower().split())
    overlap = len(prev_words & new_words)
    
    # 主语连贯性检查
    prev_first = prev_clean.split()[0].lower() if prev_clean.split() else ""
    new_first = new_clean.split()[0].lower() if new_clean.split() else ""
    subject_continuous = prev_first in {'i', 'you', 'he', 'she', 'it', 'we', 'they'} and \
                       new_first in {'i', 'you', 'he', 'she', 'it', 'we', 'they'}
    
    return overlap >= 1 or subject_continuous

def is_complete_sentence(text: str) -> bool:
    """
    判断文本是否为完整句子
    
    Args:
        text: 输入文本
        
    Returns:
        是否为完整句子
    """
    # 移除空格和标点
    text = text.strip().rstrip('.!?')
    words = text.split()
    
    if len(words) < MIN_SENTENCE_LENGTH:  # 过短的句子视为不完整
        return False
        
    # 检查基本句子结构
    first_word = words[0].lower()
    
    # 1. 检查是否以常见主语开头
    common_subjects = {'i', 'you', 'he', 'she', 'it', 'we', 'they', 'there', 'this', 'that'}
    if first_word in common_subjects:
        return True
        
    # 2. 检查是否为疑问句
    question_words = {'what', 'how', 'why', 'when', 'where', 'who', 'which'}
    if first_word in question_words:
        return True
        
    # 3. 检查是否包含谓语动词
    common_verbs = {'is', 'are', 'was', 'were', 'have', 'has', 'do', 'does', 'can', 'will', 'would'}
    if any(verb in words[1:3] for verb in common_verbs):
        return True
        
    return False

def save_translation_log(text: str, is_partial: bool = False) -> Optional[str]:
    """
    增强的日志保存功能
    
    Args:
        text: 翻译文本
        is_partial: 是否为部分翻译
        
    Returns:
        保存的日志文件路径，如果失败返回None
    """
    try:
        log_dir = os.path.join(os.getcwd(), "translation_logs")
        os.makedirs(log_dir, exist_ok=True)
        
        # 日志文件路径
        date_str = datetime.now().strftime("%Y%m%d")
        log_file = os.path.join(log_dir, f"translation_{date_str}.log")
        stats_file = os.path.join(log_dir, f"stats_{date_str}.json")
        
        # 保存翻译日志
        with open(log_file, "a", encoding="utf-8") as f:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"[{timestamp}] {'(Partial) ' if is_partial else ''}{text}\n")
        
        # 更新统计信息
        if not is_partial:
            try:
                stats = {}
                if os.path.exists(stats_file):
                    with open(stats_file, 'r') as f:
                        stats = json.load(f)
                
                stats['total_translations'] = stats.get('total_translations', 0) + 1
                stats['avg_length'] = (stats.get('avg_length', 0) * (stats.get('total_translations', 1) - 1) + 
                                     len(text.split())) / stats['total_translations']
                
                with open(stats_file, 'w') as f:
                    json.dump(stats, f, indent=2)
            except Exception as e:
                logger.warning(f"统计信息更新失败: {e}")
            
        logger.debug(f"已保存日志到: {log_file}")
        return log_file
    except Exception as e:
        logger.error(f"保存日志失败: {e}")
        return None

def truncate_translation_history(history: List[str], max_length: int = MAX_TRANSLATION_HISTORY) -> List[str]:
    """
    截断翻译历史记录
    
    Args:
        history: 翻译历史列表
        max_length: 最大长度
        
    Returns:
        截断后的历史记录
    """
    if len(history) > max_length:
        return history[-max_length:]
    return history

def truncate_previous_translations(translations: List[str], max_length: int = MAX_PREVIOUS_TRANSLATIONS) -> List[str]:
    """
    截断前序翻译记录
    
    Args:
        translations: 前序翻译列表
        max_length: 最大长度
        
    Returns:
        截断后的前序翻译记录
    """
    if len(translations) > max_length:
        return translations[-max_length:]
    return translations

def clean_translation_text(text: str) -> str:
    """
    清理翻译文本
    
    Args:
        text: 原始翻译文本
        
    Returns:
        清理后的翻译文本
    """
    if not text:
        return ""
    
    # 移除多余的空白字符
    text = " ".join(text.split())
    
    # 移除特殊字符
    text = text.strip()
    
    return text

def extract_translation_summary(translations: List[str]) -> str:
    """
    提取翻译摘要
    
    Args:
        translations: 翻译列表
        
    Returns:
        翻译摘要
    """
    if not translations:
        return "无翻译记录"
    
    # 返回最近的几条翻译
    recent_translations = translations[-5:]  # 最近5条
    return "\n".join(recent_translations) 