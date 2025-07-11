import streamlit as st

st.set_page_config(page_title="关于 - BCDetection-X", layout="wide") # Collapse sidebar for this page

# --- 页面标题和简介 ---
st.title("🩸 关于 BCDetection-X")
st.markdown("---")
st.markdown(
    """
    **BCDetection-X** 是一款基于先进深度学习技术的智能血细胞检测系统。
    我们致力于提供一个**快速、便捷、智能化**的工具，辅助医疗专业人员进行诊断，并帮助用户更好地了解自身的健康状况。
    """
)
st.markdown("---") # 添加分隔线

# --- 使用列来组织内容 ---
col1, col2 = st.columns([2, 1]) # 让第一列更宽

with col1:
    st.subheader("🎯 主要功能")
    st.markdown(
        """
        *   **图像/视频分析:** 上传血细胞显微图像或视频，系统将自动进行分析。
        *   **实时摄像头检测:** 直接使用摄像头捕捉画面，进行实时的血细胞检测。
        *   **智能识别与计数:** 采用优化的 `BCdetection-YOLOv8` 模型，精准识别并计数白细胞 (WBC)、红细胞 (RBC) 和血小板 (Platelets)。
        *   **可视化图表:** 以直观的图表展示各类细胞的数量和分布。
        *   **初步健康提示:** 根据检测结果，结合医学常识，提供初步的健康风险预警（仅供参考）。
        *   **报告生成与下载:** 自动生成包含检测图像、数据和分析的报告，并支持下载。
        """
    )

with col2:
    st.subheader("🛠️ 技术栈")
    st.markdown(
        """
        *   **核心模型:** `BCdetection-YOLOv8` (基于 YOLOv8 优化)
        *   **Web 框架:** `Streamlit`
        *   **图像处理:** `OpenCV`
        *   **深度学习库:** `PyTorch (Ultralytics)`
        *   **数据可视化:** `Altair`, `Pandas`
        """
    )
    st.image("https://streamlit.io/images/brand/streamlit-logo-secondary-colormark-darktext.svg", width=200) # 添加 Streamlit Logo 示例
    # 您可以替换为自己的项目 Logo 或相关图片
    # st.image("path/to/your/logo.png", width=200)

st.markdown("---") # 添加分隔线

# --- 免责声明 ---
st.subheader("⚠️ 重要提示")
st.warning(
    """
    **免责声明:** 本系统提供的所有检测结果、细胞计数及健康风险提示**仅供初步参考和研究目的**，
    其准确性受图像质量、拍摄视野、样本制备等多重因素影响。
    **绝对不能替代**专业的临床血液检验（如血常规 CBC）和执业医师的诊断意见。
    如果您有任何健康疑虑，请务必咨询医生或前往正规医疗机构进行检查。
    """
)

# --- 页脚 (可选) ---
# st.markdown("---")
# st.markdown("<p style='text-align: center; color: grey;'>芯聚力科技（广东）有限公司 BCDetection-X Team</p>", unsafe_allow_html=True)