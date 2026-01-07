# -*- coding: utf-8 -*-
import os
from dotenv import load_dotenv

# 加载 .env 文件中的变量到环境变量中
load_dotenv()

# 导出变量供其他脚本使用
ACCESS_KEY_ID = os.getenv("ALIBABA_CLOUD_ACCESS_KEY_ID")
ACCESS_KEY_SECRET = os.getenv("ALIBABA_CLOUD_ACCESS_KEY_SECRET")
ROLE_ARN = os.getenv("ALIBABA_CLOUD_ROLE_ARN")
ROLE_SESSION_NAME = os.getenv("ALIBABA_CLOUD_ROLE_SESSION_NAME", "default-session")

SLS_PROJECT_NAME = os.getenv("SLS_PROJECT_NAME")
SLS_LOGSTORE_NAME = os.getenv("SLS_LOGSTORE_NAME")
SLS_REGION = os.getenv("SLS_REGION", "cn-qingdao")

# 自动将 AK/SK 注入系统环境变量，适配阿里云 SDK 的默认读取行为
os.environ["ALIBABA_CLOUD_ACCESS_KEY_ID"] = ACCESS_KEY_ID or ""
os.environ["ALIBABA_CLOUD_ACCESS_KEY_SECRET"] = ACCESS_KEY_SECRET or ""
os.environ["ALIBABA_CLOUD_ROLE_ARN"] = ROLE_ARN or ""