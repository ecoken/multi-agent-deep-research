from dotenv import load_dotenv,find_dotenv
import os
from langchain.chat_models import init_chat_model

# 加载配置文件
# find_dotenv() 确保找到 .env文件 递归查询当前项目文件夹
load_dotenv(find_dotenv())

# 保留旧配置名兼容课程代码，也支持通用模型配置。
model_name = os.getenv("LLM_MODEL") or os.getenv("LLM_QWEN_MAX")
model_options = {"timeout": 60, "max_retries": 1}
if "api.deepseek.com" in os.getenv("OPENAI_BASE_URL", ""):
    # 当前项目的 OpenAI 适配器未透传 reasoning_content，使用非思考模式
    # 以兼容 DeepSeek 的多轮工具调用协议。
    model_options["extra_body"] = {"thinking": {"type": "disabled"}}

model = init_chat_model(
    model=model_name,
    model_provider="openai",
    **model_options
)
