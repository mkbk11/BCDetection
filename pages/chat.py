import streamlit as st
import requests
import json # For pretty printing JSON if needed

# --- Ollama API 配置 ---
OLLAMA_BASE_URL = "http://127.0.0.1:11434"
GENERATE_ENDPOINT = "/api/generate"
MODEL_NAME = "deepseek-r1:7b" # 确保 Ollama 服务中已拉取此模型
# -------------------------

st.set_page_config(page_title="AI 助手", page_icon="🤖", layout="wide")

st.title("🤖 BCDetection-X - AI 助手")
st.caption(f"由 BCDetection-AI 模型驱动")

st.markdown("--- ")
# st.markdown("欢迎使用 AI 助手！您可以输入任何与医学、健康、疾病诊疗相关的问题，AI 将尽力为您提供信息和解答。")
st.markdown("欢迎使用 AI 助手！AI 将尽力为您提供信息和解答。")

# --- 会话状态初始化 ---
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'current_model' not in st.session_state:
    st.session_state.current_model = MODEL_NAME

# --- 模型选择 (可选) ---
# st.sidebar.header("模型设置")
# available_models = ["deepseek-r1:7b", "qwen:7b", "llama3"] # 示例模型列表
# selected_model = st.sidebar.selectbox("选择语言模型:", available_models, index=available_models.index(st.session_state.current_model) if st.session_state.current_model in available_models else 0)
# if selected_model != st.session_state.current_model:
#     st.session_state.current_model = selected_model
#     st.session_state.chat_history = [] # 模型更改时清空历史记录
#     st.rerun()

# --- API 调用函数 ---
def call_ollama_api(prompt_text, model_name):
    full_url = OLLAMA_BASE_URL + GENERATE_ENDPOINT
    payload = {
        "model": model_name,
        "prompt": prompt_text,
        "stream": False  # 设置为 False 以获取完整响应
    }
    try:
        response = requests.post(full_url, json=payload, timeout=180) # 增加超时时间
        response.raise_for_status()  # 如果请求失败则抛出 HTTPError
        # Ollama 返回的 JSON 中，实际的文本在 'response' 字段
        return response.json().get("response", "未能从 Ollama 获取有效响应。")
    except requests.exceptions.Timeout:
        st.error(f"调用 Ollama API 超时。请检查网络连接或 Ollama 服务状态。")
        return None
    except requests.exceptions.ConnectionError:
        st.error(f"无法连接到 Ollama 服务 ({OLLAMA_BASE_URL})。请确保服务正在运行。")
        return None
    except requests.exceptions.RequestException as e:
        st.error(f"调用 Ollama API 失败: {e}")
        st.warning(f"请确保 Ollama 服务正在运行，并且模型 '{model_name}' 已下载。您可以通过命令 `ollama pull {model_name}` 来下载模型。")
        return None

# --- 聊天界面 ---
# 显示聊天历史
chat_container = st.container()
with chat_container:
    for i, chat in enumerate(st.session_state.chat_history):
        if chat["role"] == "user":
            st.chat_message("user", avatar="🧑‍⚕️").write(chat["content"])
        else:
            st.chat_message("assistant", avatar="🤖").write(chat["content"])

# 用户输入
# prompt = st.chat_input(f"向 {st.session_state.current_model} 提问...")
prompt = st.chat_input(f"向 BCDetection-AI 提问...")


if prompt:
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    with chat_container:
        st.chat_message("user", avatar="🧑‍⚕️").write(prompt)
    
    with st.spinner(f"AI ({st.session_state.current_model}) 正在思考中..."):
        # 构建更丰富的上下文给模型 (可选，但推荐)
        # full_prompt_to_model = "\n".join([f"{msg['role']}: {msg['content']}" for msg in st.session_state.chat_history])
        # For simplicity, just send the last prompt, but for better conversation, send history.
        # For a more robust solution, you might want to manage the context window size.
        contextual_prompt = ""
        # 简单的上下文构建：包含最近几轮对话
        history_to_include = st.session_state.chat_history[-5:] # 例如，包含最近5条消息
        for msg in history_to_include:
            if msg["role"] == "user":
                contextual_prompt += f"用户: {msg['content']}\n"
            else:
                contextual_prompt += f"助手: {msg['content']}\n"
        # 确保最后是当前用户的问题
        if not contextual_prompt.endswith(f"用户: {prompt}\n"):
             contextual_prompt += f"用户: {prompt}\n"
        contextual_prompt += "助手: " # 提示模型继续回答

        ai_response = call_ollama_api(contextual_prompt, st.session_state.current_model)
        
    if ai_response:
        st.session_state.chat_history.append({"role": "assistant", "content": ai_response})
        with chat_container:
            st.chat_message("assistant", avatar="🤖").write(ai_response)
    else:
        # 如果 API 调用失败，也记录一个错误信息到聊天历史，并显示
        error_message = "抱歉，AI 助手当前无法响应。请稍后再试或检查 Ollama 服务。"
        st.session_state.chat_history.append({"role": "assistant", "content": error_message})
        with chat_container:
            st.chat_message("assistant", avatar="🤖").error(error_message)

# --- 清空聊天记录按钮 ---
if st.sidebar.button("清除聊天记录"):
    st.session_state.chat_history = []
    st.rerun()

st.sidebar.markdown("--- ")
st.sidebar.info("提示：此 AI 助手提供的信息仅供参考，不能替代专业医疗建议。")

# st.markdown("--- ")
# st.markdown("<p style='text-align: center; color: grey;'>由 芯聚力科技 提供技术支持</p>", unsafe_allow_html=True)