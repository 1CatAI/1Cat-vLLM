# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""API-like first-seen prompt corpus for the fresh-text decode metric (never reused as warm-up)."""

WARMUP = [
    "你好，今天天气怎么样？",
    "Write a haiku about the sea.",
    "用一句话解释什么是机器学习。",
    "List three fruits that are red.",
]
FRESH = [
    ("zh-history", "简要介绍一下唐朝的科举制度，以及它对后世的影响。"),
    (
        "en-science",
        "Explain how photosynthesis works, step by step, for a high school student.",
    ),
    ("zh-advice", "我下个月要去面试一家互联网公司的产品经理岗位，应该怎么准备？"),
    (
        "en-email",
        "Draft a polite email to a client explaining that the project delivery will be delayed by two weeks.",
    ),
    (
        "code-js",
        "Write a JavaScript function that debounces another function, with comments.",
    ),
    (
        "code-sql",
        "写一个 SQL 查询：找出每个部门工资最高的三名员工，表结构为 employees(id, name, dept_id, salary)。",
    ),
    (
        "code-rust",
        "Implement a thread-safe counter in Rust using Arc and Mutex, and explain the ownership rules involved.",
    ),
    (
        "code-debug",
        "下面的 Python 代码为什么会报错？如何修复？\nnums = [1, 2, 3]\nfor i in range(len(nums) + 1):\n    print(nums[i])",
    ),
    ("math-proof", "证明：任意两个奇数之和是偶数。"),
    (
        "math-word",
        "A train leaves at 9:00 traveling 80 km/h; another leaves the same station at 10:00 at 100 km/h on the same track. When does the second catch up?",
    ),
    ("zh-poem", "以“春夜”为题写一首七言绝句，并解释每一句的意境。"),
    (
        "en-story",
        "Write a short science-fiction story about a robot who learns to paint, about 300 words.",
    ),
    ("zh-story", "写一个关于外婆家老槐树的温情小故事，约300字。"),
    (
        "translate-en",
        "Translate into Chinese: The rapid development of renewable energy is reshaping the global economy, creating new industries while challenging traditional ones.",
    ),
    (
        "translate-zh",
        "把这句话翻译成日语和法语：学习一门新语言最重要的是坚持每天练习。",
    ),
    (
        "json-api",
        "Design a JSON schema for a REST API response that returns a paginated list of orders, with an example.",
    ),
    (
        "yaml-config",
        "写一个 docker-compose.yml，包含 nginx、一个 Python Flask 应用和 PostgreSQL 数据库，并加上注释。",
    ),
    ("zh-econ", "通货膨胀对普通家庭有哪些影响？普通人可以如何应对？"),
    ("en-compare", "Compare TCP and UDP in a table and explain when to use each."),
    ("zh-health", "长期熬夜对身体有哪些危害？给出改善作息的具体建议。"),
    (
        "en-philosophy",
        "What is the trolley problem, and what do utilitarian and deontological ethics say about it?",
    ),
    ("zh-travel", "帮我规划一个北京三日游的行程，包括交通和美食推荐。"),
    (
        "en-summary",
        "Summarize the main causes and consequences of the 2008 financial crisis in five bullet points.",
    ),
    ("zh-tech", "解释一下 Transformer 模型中的注意力机制，最好举一个直观的例子。"),
]
