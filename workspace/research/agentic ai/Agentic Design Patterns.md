# Agentic Design Patterns

Instead of using zero-shot mode in LLMs which is just prompting a model to generate final output token by token without revising its work, we can use agentic workflow. The difference between two approaches are shown clearly in the graph below.
<p align ="left">
    <img src="The-Batch-ads-and-exclusive-banners.png" alt="Tensor-Cores" width="600"/>
    <br />
    <span style="font-size:1em;margin-left: 60px;">Figure 1: Comparison between Turing Tensor Cores and Pascal Architecture.</span>
</p>

According to the framework for categorizing design patterns for building agents described in [Andrew Ng's blog](https://www.deeplearning.ai/the-batch/how-agents-can-improve-llm-performance/), there are four subcategories/design patterns tha can be used, either individually or together, when designing an agentic flow:

* **Reflection**: The LLM examines its own work to come up with ways to improve it. 
* **Tool Use**: The LLM is given tools such as web search, code execution, or any other function to help it gather information, take action, or process data.
* **Planning**: The LLM comes up with, and executes, a multistep plan to achieve a goal (for example, writing an outline for an essay, then doing online research, then writing a draft, and so on).
* **Multi-agent** collaboration: More than one AI agent work together, splitting up tasks and discussing and debating ideas, to come up with better solutions than a single agent would.

Blog links for further read in each design categories:
* [Agentic Design Patterns Part 2, Reflection](https://www.deeplearning.ai/the-batch/agentic-design-patterns-part-2-reflection/?ref=dl-staging-website.ghost.io)
* [Agentic Design Patterns Part 3, Tool Use](https://www.deeplearning.ai/the-batch/agentic-design-patterns-part-3-tool-use/)
* [Agentic Design Patterns Part 4, Planning](https://www.deeplearning.ai/the-batch/agentic-design-patterns-part-4-planning/)
* [Agentic Design Patterns Part 5, Multi-Agent Collaboration](https://www.deeplearning.ai/the-batch/agentic-design-patterns-part-5-multi-agent-collaboration/)


Open-Source Repos and Agentic Tools

* [Gorilla: Large Language Model Connected with Massive APIs](https://github.com/ShishirPatil/gorilla)
* [ThinkGPT](https://github.com/jina-ai/thinkgpt/tree/main)
* [Self-Refine: Iterative Refinement with Self-Feedback](https://github.com/madaan/self-refine)
* [MM-REACT](https://github.com/microsoft/MM-REACT)

Open-Source Projects Related to Our Goal
* STORM - Research Tool von Stanford
    * [Github Repo](https://github.com/stanford-oval/storm/)
    * [Demo](https://storm.genie.stanford.edu/)