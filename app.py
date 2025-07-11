import streamlit as st
from PIL import Image # 导入 Image 用于处理 Logo

st.set_page_config(
    page_title="BCDetection-X 首页",
    page_icon="🩸", # 可以替换为公司 Logo 的 URL 或本地路径
    layout="wide"
)

# --- Logo 和标题 ---
col1, col2 = st.columns([1, 5]) # 调整列的比例以适应 Logo 大小

# with col1:
#     # --- 公司 Logo ---
#     # 替换为您公司 Logo 的实际路径或 URL
#     # 建议使用相对路径（如果 Logo 在项目文件夹内）或 URL
#     try:
#         # 示例：假设 logo.png 在项目根目录
#         logo_path = "image.png"
#         logo = Image.open(logo_path)
#         st.image(logo, width=200) # 调整宽度
#     except FileNotFoundError:
#         st.warning("未找到 Logo 文件 (logo.png)，请放置 Logo 文件或修改路径。")
#     except Exception as e:
#         st.error(f"加载 Logo 时出错: {e}")


# with col2:
#     st.title("🩸 BCDetection-X：基于深度学习的血细胞智能检测系统")
#     st.caption("由 xxxx 提供支持")

st.title("🩸 BCDetection-X：基于深度学习的血细胞智能检测系统")
st.caption("由 xxxx 提供支持")

st.markdown("---") # 添加分隔线

# --- 欢迎语和主要功能介绍 ---
st.subheader("欢迎使用 BCDetection-X！👋")
st.markdown(
    """
    本系统利用先进的深度学习技术，为您提供快速、智能的血细胞分析体验。
    请在左侧导航栏选择 **细胞检测** 页面开始使用：
    """
)

# 使用列展示主要功能点，更美观
feat_col1, feat_col2 = st.columns(2)
with feat_col1:
    st.markdown(
        """
        *   🔬 **图像/视频分析:** 上传血细胞样本图像或视频。
        *   📸 **实时摄像头检测:** 利用摄像头进行即时分析。
        """
    )
with feat_col2:
    st.markdown(
        """
        *   📊 **智能计数与统计:** 自动识别并统计各类细胞。
        *   ⚠️ **初步健康提示:** 获取基于计数的分析提示 (仅供参考)。
        """
    )

st.info("您也可以选择 **关于** 页面了解更多项目详情和技术细节。")

st.markdown("---")

# --- 公司介绍模板 ---
st.subheader("关于我们 - xxxx")
# 使用列来放置公司介绍文本和可能的图片/联系方式
intro_col1, intro_col2 = st.columns([3, 1])
with intro_col1:
    st.markdown(
        """
        **核心优势:**
        *   强大的研发团队： 汇聚了人工智能、大数据、云计算等领域的资深专家和创新人才，具备持续的技术创新能力和深厚的研发实力，
        能够快速响应市场需求并提供定制化的解决方案。
        *   领先的算法模型： 掌握行业领先的深度学习、自然语言处理、计算机视觉等核心算法，并结合丰富的行业数据进行模型优化和迭代，
        确保技术方案的先进性和有效性，为客户提供更精准、高效的智能化服务。
        *   丰富的行业经验与实践： 凭借在智慧城市、智慧医疗、智慧园区等多个领域的成功案例积累了丰富的行业Know-how和项目实施经验，
        能够深刻理解行业痛点与需求，提供成熟可靠的整体解决方案，并具备跨行业应用的拓展能力。

        BCDetection-X 是我们团队精心打造的产品，旨在 赋能医疗专业人员，提升诊断效率，助力疾病的早期筛查与监测，
        并为科研人员提供强大的数据分析工具，加速血液学研究的进展。。
        """
    )
with intro_col2:
    # st.image("path/to/company/image.jpg")
    st.markdown(
        """
        **联系我们:**
        *   **地址:** xxxx
        *   **邮箱:** xxxx
        *   **电话:** xxxx
        """
    )

st.sidebar.success("请选择上方的一个页面开始。")

# --- 页脚 ---
st.markdown("--- ")
st.markdown(f"© {st.session_state.get('company_name', 'xxxxx')} BCDetection-X 团队") # 使用变量或直接修改