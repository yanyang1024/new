import json
import os
import time
from datetime import datetime
from ..config.translation_config import *

def format_translation(text, is_partial=False):
    """格式化翻译文本，添加标点符号"""
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
    question_words = COMPLETE_SENTENCE_CONFIG['question_words']
    if words[0].lower() in question_words:
        return text + "?"
    elif any(word.lower() in {'can', 'could', 'would', 'will', 'shall'} for word in words[:2]):
        return text + "?"
    else:
        return text + "."

def should_merge_translations(new_text, prev_text, last_sentence_end, test_mode=False):
    """增强的翻译合并判断逻辑"""
    if not prev_text or not new_text:
        return False
        
    # 检查时间间隔
    current_time = time.time()
    if current_time - last_sentence_end > MERGE_CONFIG['time_threshold']:
        return False
        
    # 去除标点后再判断
    prev_clean = prev_text.rstrip('.!?').strip()
    new_clean = new_text.strip()
    
    # 检查句子完整性
    if len(prev_clean.split()) < COMPLETE_SENTENCE_CONFIG['min_words']:  # 前一句过短，可能不完整
        return True
        
    # 计算词重叠
    prev_words = set(prev_clean.lower().split())
    new_words = set(new_clean.lower().split())
    overlap = len(prev_words & new_words)
    
    # 主语连贯性检查
    if MERGE_CONFIG['subject_continuity']:
        prev_first = prev_clean.split()[0].lower()
        new_first = new_clean.split()[0].lower()
        subject_continuous = prev_first in COMPLETE_SENTENCE_CONFIG['common_subjects'] and \
                           new_first in COMPLETE_SENTENCE_CONFIG['common_subjects']
    else:
        subject_continuous = False
    
    return overlap >= MERGE_CONFIG['min_overlap'] or subject_continuous

def is_complete_sentence(text):
    """判断文本是否为完整句子"""
    # 移除空格和标点
    text = text.strip().rstrip('.!?')
    words = text.split()
    
    if len(words) < COMPLETE_SENTENCE_CONFIG['min_words']:  # 过短的句子视为不完整
        return False
        
    # 检查基本句子结构
    first_word = words[0].lower()
    
    # 1. 检查是否以常见主语开头
    if first_word in COMPLETE_SENTENCE_CONFIG['common_subjects']:
        return True
        
    # 2. 检查是否为疑问句
    if first_word in COMPLETE_SENTENCE_CONFIG['question_words']:
        return True
        
    # 3. 检查是否包含谓语动词
    if any(verb in words[1:3] for verb in COMPLETE_SENTENCE_CONFIG['common_verbs']):
        return True
        
    return False

def save_translation_log(text, is_partial=False, test_mode=False):
    """增强的日志保存功能"""
    try:
        log_dir = os.path.join(os.getcwd(), LOG_CONFIG['log_dir'])
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
                if test_mode:
                    print(f"[WARNING] 统计信息更新失败: {e}")
            
        if test_mode:
            print(f"[DEBUG] 已保存日志到: {log_file}")
        return True
    except Exception as e:
        if test_mode:
            print(f"[ERROR] 保存日志失败: {e}")
        return False

def clean_translation_text(text):
    """清理翻译文本"""
    if not text:
        return ""
    
    # 移除多余的空白字符
    text = ' '.join(text.split())
    
    # 移除特殊字符
    text = text.strip()
    
    return text

def validate_translation_result(text, test_mode=False):
    """验证翻译结果的有效性"""
    if not text or text.strip() == "":
        if test_mode:
            print("[DEBUG] 翻译结果为空")
        return False
    
    # 检查是否包含有效字符
    if not any(c.isalpha() for c in text):
        if test_mode:
            print("[DEBUG] 翻译结果不包含字母")
        return False
    
    # 检查长度是否合理
    if len(text.strip()) < 2:
        if test_mode:
            print("[DEBUG] 翻译结果过短")
        return False
    
    return True

def merge_translation_texts(texts, separator=" "):
    """合并多个翻译文本"""
    if not texts:
        return ""
    
    # 过滤空文本
    valid_texts = [text.strip() for text in texts if text and text.strip()]
    
    if not valid_texts:
        return ""
    
    # 合并文本
    merged = separator.join(valid_texts)
    
    # 清理结果
    return clean_translation_text(merged) 