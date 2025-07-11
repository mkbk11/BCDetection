# BCDetection-X：基于深度学习的血细胞智能检测系统
## 📋 项目概述
BCDetection-X 是一款基于先进深度学习技术的智能血细胞检测系统。该系统利用优化的 YOLOv8 模型，能够快速、准确地识别和分析血液样本中的各类细胞，包括红细胞(RBC)、白细胞(WBC)和血小板(Platelets)。

本系统旨在为医疗专业人员提供辅助诊断工具，同时也可作为教育和研究用途的血液学分析平台。

## ✨ 主要功能
- 多种输入方式 ：
  
  - 上传图像分析：支持上传血细胞显微图像进行分析
  - 上传视频分析：支持上传视频文件进行连续分析
  - 实时摄像头检测：直接使用摄像头进行实时血细胞检测
- 智能识别与计数 ：
  
  - 精准识别红细胞(RBC)、白细胞(WBC)和血小板(Platelets)
  - 自动计数各类细胞数量
  - 提供细胞分布的可视化图表
- 健康分析 ：
  
  - 根据检测结果提供初步健康风险提示（仅供参考）
  - 生成详细的分析报告
- AI 助手 ：
  
  - 内置基于大语言模型的AI助手
  - 可回答与医学、健康相关的问题
## 🛠️ 技术栈
- 核心模型 ：BCdetection-YOLOv8 (基于 YOLOv8 优化)
- Web 框架 ：Streamlit
- 图像处理 ：OpenCV
- 深度学习库 ：PyTorch (Ultralytics)
- 数据可视化 ：Altair, Pandas
- AI 对话 ：基于 Ollama API 的本地大语言模型
## 📦 安装指南
### 前提条件
- Python 3.8 或更高版本
- CUDA 支持的 GPU (推荐用于加速推理)
### 安装步骤
1. 克隆仓库：
```
git clone https://github.com/yourusername/
BCDetection-X.git
cd BCDetection-X
```
2. 安装依赖：
```
pip install -r requirements.txt
```
3. (可选) 安装 Ollama 以启用 AI 助手功能：
   - 访问 Ollama 官网 下载并安装
   - 拉取所需模型： ollama pull deepseek-r1:7b
## 🚀 使用方法
1. 启动应用：
```
streamlit run app.py
```
2. 在浏览器中访问： http://localhost:8501
3. 使用界面：
   
   - 在左侧导航栏选择功能页面（细胞检测、AI 助手、关于）
   - 在细胞检测页面，选择输入方式并上传图像/视频或启用摄像头
   - 调整侧边栏中的模型参数（如置信度阈值）
   - 查看检测结果、细胞计数和健康提示
   - 下载分析报告
## ⚠️ 免责声明
重要提示: 本系统提供的所有检测结果、细胞计数及健康风险提示 仅供初步参考和研究目的 ，其准确性受图像质量、拍摄视野、样本制备等多重因素影响。 绝对不能替代 专业的临床血液检验（如血常规 CBC）和执业医师的诊断意见。如果您有任何健康疑虑，请务必咨询医生或前往正规医疗机构进行检查。

## 📄 许可证
MIT License

## 👥 贡献者
- 芯聚力科技（广东）有限公司 BCDetection-X 团队
## 📞 联系方式
- 地址: xxxx
- 邮箱: xxxx
- 电话: 
