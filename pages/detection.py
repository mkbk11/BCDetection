import streamlit as st
from PIL import Image
import cv2
import numpy as np
from ultralytics import YOLO
import io
import tempfile
import os
import glob
import pandas as pd
import altair as alt
import zipfile
import requests

# --- Ollama API 配置 ---
OLLAMA_BASE_URL = "http://127.0.0.1:11434"
GENERATE_ENDPOINT = "/api/generate"
OLLAMA_MODEL_NAME = "deepseek-r1:7b" # 您指定的模型名称，确保 Ollama 服务中已拉取此模型
# -------------------------

st.set_page_config(page_title="细胞检测 - BCDetection-X", layout="wide")

st.title("🔬 细胞检测")

st.write("请在侧边栏配置模型和参数，然后选择输入方式进行检测。")

# --- 配置侧边栏 ---
st.sidebar.header("⚙️ 配置选项")
input_method = st.sidebar.radio("选择输入源:", ('上传图像', '上传视频', '使用摄像头'))

# --- 模型和参数配置 ---
st.sidebar.subheader("模型设置")
# Find .pt files in the 'models' subdirectory relative to the main app directory
# Streamlit pages run from the root directory
models_dir = "models"
model_files_full_path = glob.glob(os.path.join(models_dir, "*.pt"))

# Extract just the filenames for display in the selectbox, but keep full paths for loading
model_display_names = [os.path.basename(f) for f in model_files_full_path]

if not model_files_full_path:
    st.sidebar.warning(f"在 '{models_dir}' 目录中未找到 .pt 模型文件。请检查路径。")
    selected_model_path = None
    model_display_names = ["无可用模型"]
    default_model_index = 0
else:
    default_model_name = 'BCdetection_YOLOv8.pt'
    default_model_index = 0
    if default_model_name in model_display_names:
        default_model_index = model_display_names.index(default_model_name)

    selected_display_name = st.sidebar.selectbox("选择 YOLO 模型:", model_display_names, index=default_model_index)
    selected_model_path = None
    for full_path in model_files_full_path:
        if os.path.basename(full_path) == selected_display_name:
            selected_model_path = full_path
            break

st.sidebar.subheader("推理参数")
confidence_threshold = st.sidebar.slider("置信度阈值 (Confidence)", 0.0, 1.0, 0.25, 0.05)
iou_threshold = st.sidebar.slider("交并比阈值 (IoU)", 0.0, 1.0, 0.45, 0.05)
# ----------------------

# --- 模型加载 ---
@st.cache_resource
def load_model(model_path):
    try:
        model = YOLO(model_path)
        return model
    except Exception as e:
        st.sidebar.error(f"加载模型 '{os.path.basename(model_path)}' 失败: {e}")
        return None

model = None
if selected_model_path:
    model = load_model(selected_model_path)
    if model:
        st.sidebar.info(f"当前模型: {os.path.basename(selected_model_path)}")
    else:
        st.sidebar.error("模型加载失败，请检查文件或选择其他模型。")
elif not model_files_full_path:
     st.sidebar.error("无可用模型加载。")
else:
     st.sidebar.error("未选择有效的模型路径。")

# --- 主内容区域 ---
# col1, col2 = st.columns(2)

# --- 健康风险提示函数 (增强版) ---
def provide_health_tips(counts):
    tips = []
    report_lines = ["--- BCDetection-X 分析报告 ---", ""]
    rbc_count = counts.get('RBC', 0)
    wbc_count = counts.get('WBC', 0)
    platelets_count = counts.get('Platelets', 0)
    total_cells = sum(counts.values())

    report_lines.append("**检测到的细胞计数:**")
    report_lines.append(f"- 红细胞 (RBC): {rbc_count}")
    report_lines.append(f"- 白细胞 (WBC): {wbc_count}")
    report_lines.append(f"- 血小板 (Platelets): {platelets_count}")
    report_lines.append(f"- 总细胞数: {total_cells}")
    report_lines.append("")

    tips.append("**初步分析说明:**")
    report_lines.append("**初步分析说明:**")

    if total_cells == 0:
        tip = "未在当前视野检测到细胞。请确保图像清晰且具有代表性。"
        tips.append(f"- {tip}")
        report_lines.append(f"- {tip}")
    else:
        if rbc_count > 0:
            tip = f"检测到红细胞 ({rbc_count} 个)。红细胞负责运输氧气。数量异常（过高或过低）可能与多种健康状况有关，如贫血或红细胞增多症。"
            tips.append(f"- {tip}")
            report_lines.append(f"- {tip}")
        if wbc_count > 0:
            tip = f"检测到白细胞 ({wbc_count} 个)。白细胞是免疫系统的重要组成部分。数量升高可能提示感染或炎症，降低则可能影响免疫力的。不同类型的白细胞（嗜中性粒细胞、淋巴细胞等）比例也很重要，本系统未做细分。"
            tips.append(f"- {tip}")
            report_lines.append(f"- {tip}")
        if platelets_count > 0:
            tip = f"检测到血小板 ({platelets_count} 个)。血小板在止血和凝血过程中起关键作用。数量异常可能影响凝血功能。"
            tips.append(f"- {tip}")
            report_lines.append(f"- {tip}")

    disclaimer = "重要提示：本分析基于当前视野内的细胞识别和计数，结果**高度依赖于图像质量和代表性**，**不能**替代标准化的临床血液检验（如全血细胞计数 CBC）和专业医生诊断。细胞的绝对数量、体积、分类和形态学评估需要实验室专业检测。请务必咨询医生以获取准确的健康评估。"
    tips.append(f"- {disclaimer}")
    report_lines.append("")
    report_lines.append(disclaimer)
    report_lines.append("")
    report_lines.append("--- 报告结束 ---")

    return tips, "\n".join(report_lines)

# --- 创建下载 ZIP 文件 --- #
def create_download_zip(image_np, counts, report_text):
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'a', zipfile.ZIP_DEFLATED, False) as zip_file:
        # 1. 添加分析报告文本
        zip_file.writestr('analysis_report.txt', report_text)

        # 2. 添加标注后的图像
        # 将 NumPy 图像转为 PNG 格式的字节流
        is_success, buffer = cv2.imencode(".png", image_np)
        if is_success:
            image_bytes = io.BytesIO(buffer)
            zip_file.writestr('annotated_image.png', image_bytes.getvalue())
        else:
            # 如果图像编码失败，可以写入一个错误信息或跳过
            zip_file.writestr('image_error.txt', '无法编码标注后的图像。')

    zip_buffer.seek(0)
    return zip_buffer

# --- 新增：调用 Ollama API 的函数 ---
def call_ollama_api(prompt_text):
    """调用本地 Ollama API 获取大模型分析"""
    full_url = OLLAMA_BASE_URL + GENERATE_ENDPOINT
    payload = {
        "model": OLLAMA_MODEL_NAME,
        "prompt": prompt_text,
        "stream": False  # 设置为 False 以获取完整响应
    }
    try:
        response = requests.post(full_url, json=payload, timeout=120) # 设置超时
        response.raise_for_status()  # 如果请求失败则抛出 HTTPError
        # Ollama 返回的 JSON 中，实际的文本在 'response' 字段
        return response.json().get("response", "未能从 Ollama 获取有效响应。")
    except requests.exceptions.RequestException as e:
        st.error(f"调用 Ollama API 失败: {e}")
        st.warning("请确保 Ollama 服务正在运行，并且模型 '{OLLAMA_MODEL_NAME}' 已下载。您可以通过命令 `ollama pull {OLLAMA_MODEL_NAME}` 来下载模型。")
        return None
# ------------------------------------

# --- 输入处理逻辑 ---
st.subheader("🖼️ 输入预览")
if input_method == '上传图像':
    uploaded_image = st.file_uploader("选择一个图像文件 (JPG, PNG)", type=["jpg", "png"], key="image_uploader")
    if uploaded_image is not None:
        # 保存原始图像用于后续显示
        image_bytes = uploaded_image.read()
        pil_image = Image.open(io.BytesIO(image_bytes))
        img_np = np.array(pil_image)
        if len(img_np.shape) == 3 and img_np.shape[2] == 4:
            img_np = cv2.cvtColor(img_np, cv2.COLOR_RGBA2RGB)
        elif len(img_np.shape) == 2:
            img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
        
        # 保存原始图像到session state
        st.session_state['original_image'] = img_np
        
        if model:
            # 使用已经处理过的图像，不需要重复读取和处理

            results = model.predict(img_np, conf=confidence_threshold, iou=iou_threshold)
            annotated_img = results[0].plot()
            annotated_img_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)

            counts = {'WBC': 0, 'RBC': 0, 'Platelets': 0}
            class_names = model.names
            detected_boxes = results[0].boxes
            if detected_boxes is not None and len(detected_boxes) > 0:
                detected_classes = detected_boxes.cls.cpu().numpy().astype(int)
                for i, cls_index in enumerate(detected_classes):
                    class_name = class_names.get(cls_index, f'未知类别_{cls_index}')
                    counts[class_name] = counts.get(class_name, 0) + 1
            else:
                st.write("未检测到任何目标。")

            st.session_state['annotated_image'] = annotated_img_rgb
            st.session_state['cell_counts'] = counts
            st.session_state['input_type'] = 'image' # Mark input type
            # Generate analysis text immediately after detection for image
            _, report_text = provide_health_tips(counts)
            st.session_state['report_text'] = report_text
            st.info("图像检测完成！结果见下方。")
        else:
            st.error("无法执行检测，因为模型加载失败。")
    else:
        if 'annotated_image' in st.session_state: del st.session_state['annotated_image']
        if 'cell_counts' in st.session_state: del st.session_state['cell_counts']
        if 'input_type' in st.session_state: del st.session_state['input_type']

elif input_method == '上传视频':
    uploaded_video = st.file_uploader("选择一个视频文件", type=["mp4", "avi", "mov"], key="video_uploader")
    if uploaded_video is not None:
        st.video(uploaded_video)
        st.info("视频上传成功，点击下方按钮开始处理...")
        process_video_button = st.button("处理视频")

        if process_video_button and model:
            with st.spinner('正在处理视频，请稍候...'):
                tfile = tempfile.NamedTemporaryFile(delete=False)
                tfile.write(uploaded_video.read())
                video_path = tfile.name
                tfile.close()

                cap = cv2.VideoCapture(video_path)
                if not cap.isOpened():
                    st.error(f"无法打开视频文件: {video_path}")
                    if os.path.exists(video_path): os.unlink(video_path)
                else:
                    stframe = st.empty()
                    frame_count = 0
                    total_counts = {'WBC': 0, 'RBC': 0, 'Platelets': 0}
                    class_names = model.names
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    if total_frames <= 0: st.warning("无法获取视频总帧数，进度条可能不准确。")

                    while True:
                        ret, frame = cap.read()
                        if not ret: break
                        frame_count += 1
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        results = model.predict(frame_rgb, conf=confidence_threshold, iou=iou_threshold, verbose=False)
                        annotated_frame = results[0].plot()
                        detected_boxes = results[0].boxes
                        if detected_boxes is not None and len(detected_boxes) > 0:
                            detected_classes = detected_boxes.cls.cpu().numpy().astype(int)
                            for cls_index in detected_classes:
                                class_name = class_names.get(cls_index, f'未知类别_{cls_index}')
                                total_counts[class_name] = total_counts.get(class_name, 0) + 1

                        annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                        caption_text = f'处理中 - 帧 {frame_count}'
                        if total_frames > 0: caption_text += f'/{total_frames}'
                        stframe.image(annotated_frame_rgb, caption=caption_text, use_container_width=True)
                        if total_frames > 0:
                            progress = frame_count / total_frames
                            progress_bar.progress(progress)
                            status_text.text(f"处理进度: {int(progress * 100)}%")

                    cap.release()
                    if os.path.exists(video_path): os.unlink(video_path)
                    if total_frames > 0: progress_bar.progress(1.0)
                    status_text.text(f"视频处理完成！共处理 {frame_count} 帧。")
                    st.success("视频处理完成！结果见下方。")
                    st.session_state['video_processed'] = True
                    st.session_state['video_total_counts'] = total_counts
                    st.session_state['input_type'] = 'video' # Mark input type
        elif process_video_button and not model:
             st.error("无法执行检测，因为模型加载失败。")
    else:
        if 'video_processed' in st.session_state: del st.session_state['video_processed']
        if 'video_total_counts' in st.session_state: del st.session_state['video_total_counts']
        if 'input_type' in st.session_state: del st.session_state['input_type']

elif input_method == '使用摄像头':
    st.info("点击下方按钮启动/停止摄像头检测。")
    col_cam_btn1, col_cam_btn2 = st.columns(2)
    start_button = col_cam_btn1.button('启动摄像头', key='start_cam')
    stop_button = col_cam_btn2.button('停止检测', key='stop_cam')
    stframe_cam = st.empty()

    if 'cam_running' not in st.session_state:
        st.session_state['cam_running'] = False
    if 'cam_total_counts' not in st.session_state:
        st.session_state['cam_total_counts'] = {'WBC': 0, 'RBC': 0, 'Platelets': 0}
        st.session_state['cam_running'] = False

    if start_button:
        st.session_state['cam_running'] = True
        # Reset counts when starting
        st.session_state['cam_total_counts'] = {'WBC': 0, 'RBC': 0, 'Platelets': 0}
        st.session_state['cam_running'] = True
    if stop_button:
        st.session_state['cam_running'] = False
        stframe_cam.empty()
        # Generate report when stopping camera
        if 'cam_total_counts' in st.session_state:
            _, report_text_cam = provide_health_tips(st.session_state['cam_total_counts'])
            st.session_state['report_text_cam'] = report_text_cam
        st.info("摄像头检测已停止。")

    if st.session_state['cam_running']:
        if model:
            # Initialize camera only when running
            if 'cap_cam' not in st.session_state or st.session_state['cap_cam'] is None:
                st.session_state['cap_cam'] = cv2.VideoCapture(0)
                if not st.session_state['cap_cam'].isOpened():
                    st.error("无法打开摄像头。请确保摄像头已连接并授权访问。")
                    st.session_state['cam_running'] = False # Stop if cannot open
                    st.session_state['cap_cam'] = None
                else:
                     st.info("摄像头已启动，正在进行实时检测...")

            cap_cam = st.session_state.get('cap_cam')
            if cap_cam and cap_cam.isOpened():
                ret_cam, frame_cam = cap_cam.read()
                if not ret_cam:
                    st.warning("无法从摄像头读取帧。")
                    st.session_state['cam_running'] = False # Stop if reading fails
                else:
                    frame_rgb_cam = cv2.cvtColor(frame_cam, cv2.COLOR_BGR2RGB)
                    results_cam = model.predict(frame_rgb_cam, conf=confidence_threshold, iou=iou_threshold, verbose=False)
                    annotated_frame_cam = results_cam[0].plot()
                    annotated_frame_rgb_cam = cv2.cvtColor(annotated_frame_cam, cv2.COLOR_BGR2RGB)
                    # Update counts for camera
                    detected_boxes_cam = results_cam[0].boxes
                    if detected_boxes_cam is not None and len(detected_boxes_cam) > 0:
                        detected_classes_cam = detected_boxes_cam.cls.cpu().numpy().astype(int)
                        class_names_cam = model.names
                        for cls_index_cam in detected_classes_cam:
                            class_name_cam = class_names_cam.get(cls_index_cam, f'未知类别_{cls_index_cam}')
                            st.session_state['cam_total_counts'][class_name_cam] = st.session_state['cam_total_counts'].get(class_name_cam, 0) + 1

                    stframe_cam.image(annotated_frame_rgb_cam, caption='实时检测', use_container_width=True)
                    # Trigger rerun to process next frame
                    st.rerun()
        else:
            st.error("无法启动摄像头检测，因为模型加载失败。")
            st.session_state['cam_running'] = False
    else:
         # Release camera when not running
         if 'cap_cam' in st.session_state and st.session_state['cap_cam'] is not None:
             st.session_state['cap_cam'].release()
             st.session_state['cap_cam'] = None
         # Don't clear report text here, keep it for display after stop

# --- 显示结果 ---
st.subheader("📊 检测结果与分析")

current_input_type = st.session_state.get('input_type', None)

if current_input_type == 'image' and 'annotated_image' in st.session_state:
    original_image = st.session_state['original_image']
    annotated_image = st.session_state['annotated_image']
    counts = st.session_state.get('cell_counts', {})
    report_text_internal = st.session_state.get('report_text', "报告生成失败。") # BCDetection-X 内部报告

    # 创建两列布局，左侧显示原始图像，右侧显示检测结果
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(original_image, caption='原始图像', use_container_width=True)
    
    with col2:
        st.image(annotated_image, caption='检测结果', use_container_width=True)
    st.write("**细胞计数:**")
    if counts:
        for cell_type, count in counts.items():
            st.write(f"- {cell_type}: {count}")

        # --- 添加图表 --- #
        st.subheader("细胞数量分布")
        df_counts = pd.DataFrame(list(counts.items()), columns=['细胞类型', '数量'])
        if not df_counts.empty:
            chart = alt.Chart(df_counts).mark_bar().encode(
                x=alt.X('细胞类型', sort='-y'), # Sort bars by count
                y='数量',
                color='细胞类型',
                tooltip=['细胞类型', '数量']
            ).interactive()
            st.altair_chart(chart, use_container_width=True)
        else:
            st.write("无数据显示。")

        # --- 健康风险提示 --- #
        st.subheader("⚠️ 初步分析与提示 (BCDetection-X)")
        health_tips_display, _ = provide_health_tips(counts) # provide_health_tips 返回 (tips_for_display, full_report_text)
        if health_tips_display:
            for tip in health_tips_display:
                st.warning(tip)
        
        # --- 新增：Ollama AI 分析按钮和结果展示 ---
        st.markdown("---")
        st.subheader(f"🤖 AI 大模型深度分析 (模型: BCDetection-X AI)")
        if st.button("获取 AI 进一步分析建议", key="ollama_image_analysis"):
            with st.spinner(f"正在请求 BCDetection-X AI 进行分析，请稍候..."):
                # 构建给 Ollama 的 Prompt
                prompt_to_ollama = f"""
                一名专业的血液学AI助手，请根据以下血细胞初步检测结果和分析，提供更深层次的解读、可能的病因分析以及相关的健康建议。

                [初步检测数据]
                细胞计数:
                - 白细胞 (WBC): {counts.get('WBC', 0)}
                - 红细胞 (RBC): {counts.get('RBC', 0)}
                - 血小板 (Platelets): {counts.get('Platelets', 0)}
                初步分析与提示 (来自 BCDetection-X 系统):
                {report_text_internal}

                [你的任务]
                1.  基于上述数据，详细分析各种细胞计数的临床意义。
                2.  探讨这些数据组合可能指向的潜在健康问题或疾病风险（例如，贫血、感染、炎症、凝血功能异常等）。
                3.  如果数据存在异常，请推测可能的病因。
                4.  提供一些常规的健康管理建议或后续检查建议。
                5.  请用清晰、易懂的语言进行解释，并分点阐述。
                6.  在回答的最后，请务必强调：“以上分析仅为AI模型根据提供数据进行的推测，不能替代专业医生的诊断。如有任何健康疑虑，请务必咨询执业医师。”

                请开始你的分析：
                """
                ollama_response = call_ollama_api(prompt_to_ollama)
                if ollama_response:
                    st.session_state['ollama_image_report'] = ollama_response
                else:
                    st.session_state['ollama_image_report'] = "AI 分析失败或无响应。"
        
        if 'ollama_image_report' in st.session_state:
            with st.expander("查看 AI 大模型分析结果", expanded=True):
                st.markdown(st.session_state['ollama_image_report'])
        # ------------------------------------------

        # --- 下载按钮 --- #
        # ... (下载按钮代码保持不变, 但可以考虑将 Ollama 报告也加入 ZIP) ...

    else:
        st.write("未检测到符合条件的细胞。")

elif current_input_type == 'video' and st.session_state.get('video_processed', False):
    st.write("**视频处理完成。**")
    st.write("**总细胞计数 (估算):**")
    total_counts_video = st.session_state.get('video_total_counts', {})
    # 假设视频也有一个初步报告文本，如果之前没有生成，这里需要生成或调整
    # _, report_text_video_internal = provide_health_tips(total_counts_video) # 示例
    # st.session_state['report_text_video_internal'] = report_text_video_internal

    if total_counts_video:
        for cell_type, count in total_counts_video.items():
             st.write(f"- {cell_type}: {count}")

        # ... (视频的图表和初步健康提示代码保持不变) ...
        
        # --- 新增：Ollama AI 分析按钮和结果展示 (视频) ---
        st.markdown("---")
        st.subheader(f"🤖 AI 大模型深度分析 (模型: {OLLAMA_MODEL_NAME})")
        if st.button("获取 AI 进一步分析建议 (视频)", key="ollama_video_analysis"):
            with st.spinner(f"正在请求 BCDetection-X AI 进行分析，请稍候..."):
                # 为了演示，我们假设视频也有一个 report_text_internal
                # 在实际应用中，您可能需要为视频结果单独生成或汇总一个初步报告
                video_preliminary_report = "视频初步分析摘要：\n" + \
                                           f"- 白细胞 (WBC) 总计: {total_counts_video.get('WBC', 0)}\n" + \
                                           f"- 红细胞 (RBC) 总计: {total_counts_video.get('RBC', 0)}\n" + \
                                           f"- 血小板 (Platelets) 总计: {total_counts_video.get('Platelets', 0)}\n" + \
                                           "请注意，视频结果为多帧累积估算。"

                prompt_to_ollama_video = f"""
                一名专业的血液学AI助手，请根据以下视频中血细胞的累积估算结果和分析，提供更深层次的解读、可能的病因分析以及相关的健康建议。

                [初步检测数据 - 视频估算]
                细胞总计数 (估算):
                - 白细胞 (WBC): {total_counts_video.get('WBC', 0)}
                - 红细胞 (RBC): {total_counts_video.get('RBC', 0)}
                - 血小板 (Platelets): {total_counts_video.get('Platelets', 0)}
                初步分析与提示 (来自 BCDetection-X 系统 - 视频摘要):
                {video_preliminary_report} 

                [你的任务]
                1.  基于上述累积估算数据，分析各种细胞计数的临床意义。
                2.  探讨这些数据组合可能指向的潜在健康问题或疾病风险。
                3.  如果数据存在异常，请推测可能的病因。
                4.  提供一些常规的健康管理建议或后续检查建议。
                5.  请用清晰、易懂的语言进行解释，并分点阐述。
                6.  在回答的最后，请务必强调：“以上分析仅为AI模型根据提供数据进行的推测，且基于视频多帧估算，不能替代专业医生的诊断。如有任何健康疑虑，请务必咨询执业医师。”

                请开始你的分析：
                """
                ollama_response_video = call_ollama_api(prompt_to_ollama_video)
                if ollama_response_video:
                    st.session_state['ollama_video_report'] = ollama_response_video
                else:
                    st.session_state['ollama_video_report'] = "AI 分析失败或无响应。"
            
            if 'ollama_video_report' in st.session_state:
                with st.expander("查看 AI 大模型分析结果 (视频)", expanded=True):
                    st.markdown(st.session_state['ollama_video_report'])
            # ------------------------------------------

            # --- 下载按钮 (文本报告) --- #
            # ... (下载按钮代码保持不变) ...
        else:
             st.write("在视频中未检测到符合条件的细胞。")

    elif input_method == '使用摄像头':
        if st.session_state.get('cam_running', False):
            st.info("实时检测结果显示在左侧，统计图表在下方实时更新。")
            st.subheader("实时细胞计数 (累积)")
            cam_counts = st.session_state.get('cam_total_counts', {})
            if cam_counts and sum(cam_counts.values()) > 0:
                df_counts_cam = pd.DataFrame(list(cam_counts.items()), columns=['细胞类型', '数量'])
                chart_cam = alt.Chart(df_counts_cam).mark_bar().encode(
                    x=alt.X('细胞类型', sort='-y'),
                    y='数量',
                    color='细胞类型',
                    tooltip=['细胞类型', '数量']
                ).interactive()
                st.altair_chart(chart_cam, use_container_width=True)
            else:
                st.write("摄像头运行中，暂未检测到细胞或计数为零。")
        else:
            # Display final counts after stopping
            st.info("摄像头已停止。显示最终累积计数：")
            final_cam_counts = st.session_state.get('cam_total_counts', {})
            report_text_cam = st.session_state.get('report_text_cam', "报告生成失败。")

            if final_cam_counts and sum(final_cam_counts.values()) > 0:
                st.write("**最终细胞计数 (摄像头):**")
                for cell_type, count in final_cam_counts.items():
                    st.write(f"- {cell_type}: {count}")

                st.subheader("最终细胞数量分布 (摄像头)")
                df_final_counts_cam = pd.DataFrame(list(final_cam_counts.items()), columns=['细胞类型', '数量'])
                chart_final_cam = alt.Chart(df_final_counts_cam).mark_bar().encode(
                    x=alt.X('细胞类型', sort='-y'),
                    y='数量',
                    color='细胞类型',
                    tooltip=['细胞类型', '数量']
                ).interactive()
                st.altair_chart(chart_final_cam, use_container_width=True)

                # Health tips based on final count
                st.subheader("⚠️ 初步分析与提示 (基于摄像头累积计数，仅供参考)")
                health_tips_cam, _ = provide_health_tips(final_cam_counts)
                if health_tips_cam:
                    for tip in health_tips_cam:
                        st.warning(tip)
                # st.caption("免责声明: 本系统结果不能替代专业医疗诊断。")
            else:
                st.write("摄像头运行时未检测到细胞，或最终计数为零。")
        # Potential place for cumulative stats if implemented for camera

    else:
        st.info("请先选择输入源并上传文件或启动摄像头以查看结果。")