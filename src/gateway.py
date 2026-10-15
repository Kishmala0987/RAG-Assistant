"""Step 6b: Routes the llm through  portkey gateway"""
import json
from langchain_openai import ChatOpenAI
from portkey_ai import createHeaders, PORTKEY_GATEWAY_URL
from src import config
from src.logger import get_logger

logger = get_logger(__name__)

"""
AI Compatible Request code
----
Authorization - api key -> header
model = ---,
message:[{
"role":"user",
"content":abc
}]
whenever trying to request an LLM service or any service,need to send request qith headers which indicaticates that if one is allow for this request or not
"""
#model routing
# PRIMARY_TARGET = {"provider":"@hrpolicy", "override_params":{
#     "model": config.LLM_MODEL_NAME,
#     "temperature": 0.3
# }}

# FALLBACK_TARGET = {"provider":"@hrpolicybackup", "override_params":{
#     "model": config.LLM_MODEL_NAME,
#     "temperature": 0.3

# }}
#gateway features
# load balancing
# routing
# caching
#fallback

#To get these features you have to write configuration (can be done on platform or here in the code)
#Two typpes of cofig 1)Public 2)Private

# GATEWAY_CONFIG = {
#     "strategy":{
#         "mode": "fallback"
#     },
#     "targets" : [PRIMARY_TARGET, FALLBACK_TARGET]
# }
PRIMARY_PROVIDER= "@hrpolicy"
def get_gateway_llm() -> ChatOpenAI:
    """Routing LLM cals through portkey with automatic fallback"""
    logger.info(f"Routing LLM cals through portkey with automatic fallback (primary = {PRIMARY_PROVIDER})")
    headers = createHeaders(api_key = config.PORTKEY_API_KEY, provider = "@hrpolicy")
    return ChatOpenAI(
        api_key="portkey",#it is dummy as we already sending in headers but can also write it here
        base_url=PORTKEY_GATEWAY_URL,#address of the recipient,
        model = config.LLM_MODEL_NAME, 
        default_headers=headers
    )