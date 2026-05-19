from src.llm_compiler.constants import END_OF_PLAN, JOINNER_FINISH

PLANNER_PROMPT = (
    "Question: Compare the net profit margin of Apple Inc. and Microsoft in 2023, which one is higher?\n"
    '1. search("Apple Inc.")\n'
    '2. search("Microsoft")\n'
    '3. math("Apple net profit margin in 2023 in percentage", ["$1"])\n'
    '4. math("Microsoft net profit margin in 2023 in percentage", ["$2"])\n'
    "Thought: Comparing the profit margins, Microsoft has higher margin.\n"
    f"5. join(){END_OF_PLAN}\n"
    "###\n"
    "\n"
    "Question: If Amazon's 2022 revenue increased by 10% and Google's 2022 revenue decreased by 5%, what would be the difference in their revenue in billions?\n"
    '1. search("Amazon revenue 2022")\n'
    '2. search("Alphabet Google revenue 2022")\n'
    '3. math("Amazon revenue in billions if increased by 10%", ["$1"])\n'
    '4. math("Google revenue in billions if decreased by 5%", ["$2"])\n'
    '5. math("absolute difference between $3 and $4 in billions", ["$3", "$4"])\n'
    "Thought: I can answer the question now.\n"
    f"6. join(){END_OF_PLAN}\n"
    "###\n"
    "\n"
    "Question: Calculate the total market cap of Tesla, NIO, and XPeng in 2023, and find the average.\n"
    '1. search("Tesla market cap 2023")\n'
    '2. search("NIO market cap 2023")\n'
    '3. search("XPeng market cap 2023")\n'
    '4. math("Tesla market cap in billions", ["$1"])\n'
    '5. math("NIO market cap in billions", ["$2"])\n'
    '6. math("XPeng market cap in billions", ["$3"])\n'
    '7. math("sum of $4, $5, $6 divided by 3 in billions", ["$4", "$5", "$6"])\n'
    "Thought: I can answer the question now.\n"
    f"8. join(){END_OF_PLAN}\n"
    "###\n"
    "\n"
    "Question: Compare the GDP of China and the United States in 2022, and calculate the difference as a percentage of the US GDP.\n"
    '1. search("China GDP 2022")\n'
    '2. search("United States GDP 2022")\n'
    '3. math("China GDP in trillions", ["$1"])\n'
    '4. math("United States GDP in trillions", ["$2"])\n'
    '5. math("(($4 - $3) / $4) * 100 in percentage", ["$3", "$4"])\n'
    "Thought: I can answer the question now.\n"
    f"6. join(){END_OF_PLAN}\n"
    "###\n"
    "\n"
    "Question: If Nvidia's market cap increased by 50% and AMD's increased by 30% in 2023, which company would have a higher market cap?\n"
    '1. search("Nvidia market cap 2023")\n'
    '2. search("AMD market cap 2023")\n'
    '3. math("Nvidia market cap in billions if increased by 50%", ["$1"])\n'
    '4. math("AMD market cap in billions if increased by 30%", ["$2"])\n'
    "Thought: I can compare the numbers now.\n"
    f"5. join(){END_OF_PLAN}\n"
    "###\n"
)

OUTPUT_PROMPT = (
    "Solve a financial question answering task with interleaving Observation, Thought, and Action steps. "
    "Answer should always be a single item and MUST not be multiple choices.\n"
    "Thought step can reason about the observations in a few words. You MUST keep it short.\n"
    "Action can be only one type:"
    f" (1) {JOINNER_FINISH}(answer): returns the answer and finishes the task. "
    "    - Final answer MUST NOT contain any description, and must be short (e.g. company names, numbers, Yes/No, etc.)\n"
    "    - When comparing companies or metrics, return the name of the company or the specific value asked.\n"
    "    - If you are asked about a value (e.g. ratio, difference, average, ...), it has to be a number, not a description.\n"
    "\n"
    "Here are some examples:\n"
    "\n"
    "Question: Compare the net profit margin of Apple Inc. and Microsoft in 2023, which one is higher?\n"
    "search(Apple Inc.)\n"
    "Observation: Apple Inc. is a technology company... net income $94.68 billion in 2022 on revenue of $394.33 billion...\n"
    "search(Microsoft)\n"
    "Observation: Microsoft Corporation is a technology company... net income $72.74 billion in 2022 on revenue of $198.27 billion...\n"
    "math(Apple net profit margin in 2023 in percentage)\n"
    "Observation: 25.3\n"
    "math(Microsoft net profit margin in 2023 in percentage)\n"
    "Observation: 36.4\n"
    "Thought: Microsoft has higher profit margin at 36.4% vs Apple's 25.3%.\n"
    f"Action: {JOINNER_FINISH}(Microsoft)\n"
    "###\n"
    "\n"
    "Question: If Amazon's 2022 revenue increased by 10% and Google's 2022 revenue decreased by 5%, what would be the difference in their revenue in billions?\n"
    "search(Amazon revenue 2022)\n"
    "Observation: Amazon reported revenue of $514 billion in 2022...\n"
    "search(Alphabet Google revenue 2022)\n"
    "Observation: Alphabet reported revenue of $282.8 billion in 2022...\n"
    "math(Amazon revenue in billions if increased by 10%)\n"
    "Observation: 565.4\n"
    "math(Google revenue in billions if decreased by 5%)\n"
    "Observation: 268.66\n"
    "math(absolute difference between 565.4 and 268.66 in billions)\n"
    "Observation: 296.74\n"
    "Thought: The difference is 296.74 billion.\n"
    f"Action: {JOINNER_FINISH}(296.74)\n"
    "###\n"
)
