"""
Gradio界面模块
包含用户界面相关的代码
"""

import gradio as gr
import threading
import time
import atexit
import logging

from ..core.audio_translator import AudioTranslator

logger = logging.getLogger(__name__)

def create_translation_interface():
    """创建翻译界面"""
    translator = AudioTranslator(test_mode=True)
    
    # 添加全局停止机制
    def cleanup_on_exit():
        """应用退出时的清理函数"""
        try:
            if translator.running:
                translator.close()
                logger.info("[INFO] 应用退出时清理完成")
        except Exception as e:
            logger.error(f"[ERROR] 应用退出时清理异常: {e}")
    
    # 注册退出时的清理函数
    atexit.register(cleanup_on_exit)
    
    with gr.Blocks(title="实时语音翻译系统") as demo:
        # 修改Timer配置
        update_timer = gr.Timer(value=0.3, active=False)
        
        gr.Markdown("## 实时中英语音翻译系统 (麦克风模式)")
        
        with gr.Row():
            with gr.Column(scale=1):
                status_indicator = gr.Label("系统状态: 等待启动", label="状态指示")
                with gr.Row():
                    mic_btn = gr.Button("从麦克风翻译", variant="primary")
                    stop_btn = gr.Button("停止翻译", variant="secondary")
                
                # 新增：Gradio流式麦克风组件
                gradio_mic = gr.Audio(
                    sources=["microphone"],
                    streaming=True,
                    label="实时麦克风输入",
                    visible=False
                )
            
            with gr.Column(scale=3):
                latest_translation = gr.Textbox(
                    label="最新完整翻译",
                    placeholder="等待翻译...",
                    lines=1,
                    interactive=False
                )
                
                translation_output = gr.Textbox(
                    label="实时英文字幕",
                    placeholder="等待语音输入...",
                    lines=3,
                    interactive=False
                )
                
                history_output = gr.Textbox(
                    label="翻译历史",
                    lines=5,
                    interactive=False
                )
        
        debug_output = gr.Textbox(
            label="调试日志",
            lines=10,
            interactive=False
        )

        # 状态管理
        running = gr.State(False)

        def update_components():
            """获取最新翻译数据"""
            try:
                if running.value and translator.running:
                    current_text = translator.translator.current_translation
                    latest_complete = translator.translator.latest_complete_translation or "等待完整翻译..."
                    
                    # 构建历史记录
                    history = []
                    if translator.translator.latest_complete_translation:
                        history.append(translator.translator.latest_complete_translation)
                    history.extend(translator.translator.translation_history[-4:])  # 只取最近4条
                    history_text = "\n".join(filter(None, history)) or "等待翻译..."
                    
                    # 获取调试信息
                    debug_text = translator.get_debug_info() or "等待日志..."
                    
                    # 根据输入模式显示不同状态
                    status_text = "系统状态: 运行中"
                    if translator.input_mode == 'mic':
                        status_text = "系统状态: 运行中(UDP麦克风模式)"
                    elif translator.input_mode == 'gradio_mic':
                        status_text = "系统状态: 运行中(Gradio麦克风模式)"
                        # 获取Gradio处理状态
                        gradio_status = translator.get_gradio_status()
                        if gradio_status['processing']:
                            status_text += f" - 处理中(缓冲区:{gradio_status['buffer_size']})"
                        elif gradio_status['buffer_size'] > 0:
                            status_text += f" - 等待处理(缓冲区:{gradio_status['buffer_size']})"
                    
                    return (
                        latest_complete,
                        current_text or "等待输入...",
                        history_text,
                        debug_text,
                        status_text
                    )
                return (
                    "翻译已停止",
                    "翻译已停止",
                    translator.get_all_translations() or "无历史记录",
                    translator.get_debug_info() or "无日志",
                    "系统状态: 已停止"
                )
            except Exception as e:
                logger.error(f"[ERROR] 更新异常: {e}")
                return (
                    "更新出错",
                    "更新出错",
                    "更新出错",
                    f"异常: {str(e)}",
                    "系统状态: 错误"
                )

        def start_translation_from_mic():
            """从麦克风启动翻译（Gradio流式模式）"""
            try:
                if not running.value:
                    translator.set_input_mode('gradio_mic')
                    running.value = True
                    translator.running = True
                    translator.translator.reset_state()
                    
                    threading.Thread(
                        target=translator.run,
                        daemon=True
                    ).start()
                    
                    return [
                        True,
                        "等待输入...",
                        "等待输入...",
                        "无历史记录",
                        translator.get_debug_info(),
                        "系统状态: 运行中(Gradio麦克风模式)",
                        gr.Timer(value=0.3, active=True),
                        gr.update(visible=True)  # 显示Gradio麦克风组件
                    ]
                else:
                    # 如果已经在运行，先停止再重新启动
                    translator.close()
                    running.value = False
                    time.sleep(0.5)  # 等待资源释放
                    
                    translator.set_input_mode('gradio_mic')
                    running.value = True
                    translator.running = True
                    translator.translator.reset_state()
                    
                    threading.Thread(
                        target=translator.run,
                        daemon=True
                    ).start()
                    
                    return [
                        True,
                        "等待输入...",
                        "等待输入...",
                        "无历史记录",
                        translator.get_debug_info(),
                        "系统状态: 运行中(Gradio麦克风模式)",
                        gr.Timer(value=0.3, active=True),
                        gr.update(visible=True)  # 显示Gradio麦克风组件
                    ]
            except Exception as e:
                logger.error(f"[ERROR] 启动失败: {e}")
                return [
                    False,
                    "启动失败",
                    "启动失败",
                    "启动失败",
                    f"异常: {str(e)}",
                    "系统状态: 错误",
                    gr.Timer(value=0.3, active=False),
                    gr.update(visible=False)
                ]

        def process_gradio_mic_stream(stream, new_chunk):
            """处理Gradio麦克风流式音频"""
            try:
                if not running.value or not translator.running:
                    return stream, ""
                
                # 调用translator的Gradio音频处理方法
                updated_stream, _ = translator.process_gradio_audio_stream(stream, new_chunk)
                return updated_stream, ""
                
            except Exception as e:
                logger.error(f"[ERROR] Gradio麦克风处理异常: {e}")
                return stream, ""

        def stop_translation():
            """停止翻译流程"""
            try:
                running.value = False
                translator.running = False
                translator.close()
                translator.reset_gradio_state()  # 重置Gradio状态
                summary = translator.get_all_translations()
                return [
                    "翻译已停止",
                    "翻译已停止",
                    summary or "无历史记录",
                    "无日志",
                    "系统状态: 已停止",
                    gr.Timer(value=0.3, active=False),
                    gr.update(visible=False)  # 隐藏Gradio麦克风组件
                ]
            except Exception as e:
                logger.error(f"[ERROR] 停止失败: {e}")
                return [
                    "停止失败",
                    "停止失败",
                    "停止失败",
                    f"异常: {str(e)}",
                    "系统状态: 错误",
                    gr.Timer(value=0.3, active=False),
                    gr.update(visible=False)
                ]

        # 事件绑定
        update_timer.tick(
            fn=update_components,
            outputs=[
                latest_translation,
                translation_output,
                history_output,
                debug_output,
                status_indicator
            ]
        )
        
        mic_btn.click(
            fn=start_translation_from_mic,
            outputs=[
                running,
                latest_translation,
                translation_output,
                history_output,
                debug_output,
                status_indicator,
                update_timer,
                gradio_mic
            ]
        )
        
        # 新增：Gradio麦克风流式处理事件绑定
        gradio_mic.stream(
            fn=process_gradio_mic_stream,
            inputs=[gr.State(None), gradio_mic],
            outputs=[gr.State(None), gr.Textbox()],
            show_progress=False,
            batch=False,
            max_batch_size=1
        )
        
        stop_btn.click(
            fn=stop_translation,
            outputs=[
                latest_translation,
                translation_output,
                history_output,
                debug_output,
                status_indicator,
                update_timer,
                gradio_mic
            ]
        )

    return demo

def launch_interface():
    """启动界面"""
    try:
        demo = create_translation_interface()
        demo.queue(
            default_concurrency_limit=3,
            api_open=False,
            max_size=100  # 增加队列大小以处理更多更新请求
        ).launch(
            server_name="0.0.0.0",
            server_port=7860,
            show_error=True,
            debug=True
        )
    except KeyboardInterrupt:
        logger.info("\n[INFO] 收到中断信号，正在关闭应用...")
    except Exception as e:
        logger.error(f"[ERROR] 应用启动异常: {e}")
    finally:
        logger.info("[INFO] 应用已退出") 