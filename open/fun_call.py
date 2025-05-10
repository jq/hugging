"""
https://dify.ai/  生成大模型的 应用
action 是 让openai 决策使用背后哪个 function， 然后将function结果变成自然语言输出
https://platform.openai.com/docs/actions

https://platform.openai.com/docs/guides/function-calling
是用户选择了一个function, 用自然语言调用，function 将用户输入和自己的function 给到openai
openai 从用户语言提取参数，return给 function，然后function 调用，产生的结果 再给到openai
openai 将结果转换成自然语言输出给用户

自然语言生成中关于幻觉研究的综述：https://arxiv.org/abs/2202.03629
语言模型出现的幻觉是如何滚雪球的：https://arxiv.org/abs/2305.13534
ChatGPT 在推理、幻觉和交互性上的评估：https://arxiv.org/abs/2302.04023
对比学习减少对话中的幻觉：https://arxiv.org/abs/2212.10400
自洽性提高了语言模型的思维链推理能力：https://arxiv.org/abs/2203.11171
生成式大型语言模型的黑盒幻觉检测：https://arxiv.org/abs/2303.08896

"""
# 初始化
from openai import OpenAI
from dotenv import load_dotenv, find_dotenv
import json

_ = load_dotenv(find_dotenv())

client = OpenAI()


def print_json(data):
    """
    打印参数。如果参数是有结构的（如字典或列表），则以格式化的 JSON 形式打印；
    否则，直接打印该值。
    """
    if hasattr(data, 'model_dump_json'):
        data = json.loads(data.model_dump_json())

    if (isinstance(data, (list))):
        for item in data:
            print_json(item)
    elif (isinstance(data, (dict))):
        print(json.dumps(
            data,
            indent=4,
            ensure_ascii=False
        ))
    else:
        print(data)

def get_completion(messages, model="gpt-3.5-turbo"):
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.7,  # 模型输出的随机性，0 表示随机性最小
        tools=[{  # 用 JSON 描述函数。可以定义多个。由大模型决定调用谁。也可能都不调用
            "type": "function",
            "function": {
                "name": "sum",
                "description": "加法器，计算一组数的和",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "numbers": {
                            "type": "array",
                            "items": {
                                "type": "number"
                            }
                        }
                    }
                }
            }
        }],
    )
    return response.choices[0].message

from math import *

prompt = "Tell me the sum of 1, 2, 3, 4, 5, 6, 7, 8, 9, 10."
# prompt = "桌上有 2 个苹果，四个桃子和 3 本书，一共有几个水果？"
# prompt = "1+2+3...+99+100"
# prompt = "1024 乘以 1024 是多少？"   # Tools 里没有定义乘法，会怎样？
# prompt = "太阳从哪边升起？"           # 不需要算加法，会怎样？

messages = [
    {"role": "system", "content": "你是一个数学家"},
    {"role": "user", "content": prompt}
]
response = get_completion(messages)

# 把大模型的回复加入到对话历史中
messages.append(response)

print("=====GPT回复=====")
print_json(response)

# 如果返回的是函数调用结果，则打印出来
if (response.tool_calls is not None):
    # 是否要调用 sum
    tool_call = response.tool_calls[0]
    if (tool_call.function.name == "sum"):
        # 调用 sum
        args = json.loads(tool_call.function.arguments)
        result = sum(args["numbers"])
        print("=====函数返回=====")
        print(result)

        # 把函数调用结果加入到对话历史中
        messages.append(
            {
                "tool_call_id": tool_call.id,  # 用于标识函数调用的 ID
                "role": "tool",
                "name": "sum",
                "content": str(result)  # 数值 result 必须转成字符串
            }
        )

        # 再次调用大模型
        print("=====最终回复=====")
        print(get_completion(messages).content)