from src.llm_compiler.constants import END_OF_PLAN, JOINNER_FINISH

PLANNER_PROMPT = (
    "Question: Who has a higher net worth: Elon Musk or Jeff Bezos?\n"
    '1. search("Elon Musk")\n'
    '2. search("Jeff Bezos")\n'
    "Thought: Elon Musk has higher net worth.\n"
    f"3. join(){END_OF_PLAN}\n"
    "###\n"
    "\n"
    "Question: Calculate the difference between Elon Musk's net worth and Warren Buffett's net worth in billions.\n"
    '1. search("Elon Musk")\n'
    '2. search("Warren Buffett")\n'
    '3. math("Elon Musk net worth in billions", ["$1"])\n'
    '4. math("Warren Buffett net worth in billions", ["$2"])\n'
    '5. math("difference between $3 and $4 in billions", ["$3", "$4"])\n'
    "Thought: I can answer the question now.\n"
    f"6. join(){END_OF_PLAN}\n"
    "###\n"
    "\n"
    "Question: Calculate the total net worth of Elon Musk, Jeff Bezos, and Warren Buffett in billions.\n"
    '1. search("Elon Musk")\n'
    '2. search("Jeff Bezos")\n'
    '3. search("Warren Buffett")\n'
    '4. math("Elon Musk net worth in billions", ["$1"])\n'
    '5. math("Jeff Bezos net worth in billions", ["$2"])\n'
    '6. math("Warren Buffett net worth in billions", ["$3"])\n'
    '7. math("sum of $4, $5, $6 in billions", ["$4", "$5", "$6"])\n'
    "Thought: I can answer the question now.\n"
    f"8. join(){END_OF_PLAN}\n"
    "###\n"
    "\n"
    "Question: If Elon Musk's net worth increased by 50% and Jeff Bezos's decreased by 10%, who would have a higher net worth?\n"
    '1. search("Elon Musk")\n'
    '2. search("Jeff Bezos")\n'
    '3. math("Elon Musk net worth in billions if increased by 50%", ["$1"])\n'
    '4. math("Jeff Bezos net worth in billions if decreased by 10%", ["$2"])\n'
    "Thought: Elon Musk still has higher net worth.\n"
    f"5. join(){END_OF_PLAN}\n"
    "###\n"
    "\n"
    "Question: Which company was founded earlier: Apple or Tesla?\n"
    '1. search("Apple Inc.")\n'
    '2. search("Tesla, Inc.")\n'
    "Thought: Apple was founded in 1976, Tesla in 2003. Apple is older.\n"
    f"3. join(){END_OF_PLAN}\n"
    "###\n"
)

OUTPUT_PROMPT = (
    "Solve a financial question answering task with interleaving Observation, Thought, and Action steps. "
    "Answer should always be a single item and MUST not be multiple choices.\n"
    "Thought step can reason about the observations in a few words. You MUST keep it short.\n"
    "Action can be only one type:"
    f" (1) {JOINNER_FINISH}(answer): returns the answer and finishes the task. "
    "    - Final answer MUST NOT contain any description, and must be short (e.g. person names, company names, numbers, etc.)\n"
    "    - When comparing people or companies, return the name of the person or company.\n"
    "    - If you are asked about a value (e.g. ratio, difference, average, ...), it has to be a number, not a description.\n"
    "\n"
    "Here are some examples:\n"
    "\n"
    "Question: Who has a higher net worth: Elon Musk or Jeff Bezos?\n"
    "search(Elon Musk)\n"
    "Observation: Elon Musk net worth is US$788 billion...\n"
    "search(Jeff Bezos)\n"
    "Observation: Jeff Bezos net worth is US$284.1 billion...\n"
    "Thought: Elon Musk has higher net worth at $788B vs $284.1B.\n"
    f"Action: {JOINNER_FINISH}(Elon Musk)\n"
    "###\n"
    "\n"
    "Question: Calculate the difference between Elon Musk's net worth and Warren Buffett's net worth in billions.\n"
    "search(Elon Musk)\n"
    "Observation: Elon Musk net worth is US$788 billion...\n"
    "search(Warren Buffett)\n"
    "Observation: Warren Buffett net worth is US$148.9 billion...\n"
    "math(Elon Musk net worth in billions)\n"
    "Observation: 788\n"
    "math(Warren Buffett net worth in billions)\n"
    "Observation: 148.9\n"
    "math(difference between 788 and 148.9 in billions)\n"
    "Observation: 639.1\n"
    "Thought: The difference is 639.1 billion.\n"
    f"Action: {JOINNER_FINISH}(639.1)\n"
    "###\n"
)
