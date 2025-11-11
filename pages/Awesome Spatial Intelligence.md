- ## Other Awesomes
	- ![GitHub Repo stars](https://img.shields.io/github/stars/yukangcao/Awesome-4D-Spatial-Intelligence) https://github.com/yukangcao/Awesome-4D-Spatial-Intelligence
	- ![GitHub Repo stars](https://img.shields.io/github/stars/zhengxuJosh/Awesome-Multimodal-Spatial-Reasoning) https://github.com/zhengxuJosh/Awesome-Multimodal-Spatial-Reasoning
	- ![GitHub Repo stars](https://img.shields.io/github/stars/mll-lab-nu/Awesome-Spatial-Intelligence-in-VLM) https://github.com/mll-lab-nu/Awesome-Spatial-Intelligence-in-VLM
	- ![GitHub Repo stars](https://img.shields.io/github/stars/vaew/Awesome-spatial-visual-reasoning-MLLMs) https://github.com/vaew/Awesome-spatial-visual-reasoning-MLLMs
	- ![GitHub Repo stars](https://img.shields.io/github/stars/SIBench/Awesome-Visual-Spatial-Reasoning) https://github.com/SIBench/Awesome-Visual-Spatial-Reasoning
	- ![GitHub Repo stars](https://img.shields.io/github/stars/lif314/Awesome-Spatial-Intelligence) https://github.com/lif314/Awesome-Spatial-Intelligence
	- ![GitHub Repo stars](https://img.shields.io/github/stars/arijitray1993/awesome-spatial-reasoning) https://github.com/arijitray1993/awesome-spatial-reasoning
	- ![GitHub Repo stars](https://img.shields.io/github/stars/zjwzcx/awesome-spatial-exploration-policy) https://github.com/zjwzcx/awesome-spatial-exploration-policy
	- ![GitHub Repo stars](https://img.shields.io/github/stars/layumi/Awesome-Aerial-Spatial-Intelligence) https://github.com/layumi/Awesome-Aerial-Spatial-Intelligence ==[[hrz]]==
	- ![GitHub Repo stars](https://img.shields.io/github/stars/bobochow/Awesome-Visual-Spatial-Intelligence) https://github.com/bobochow/Awesome-Visual-Spatial-Intelligence
	- ![GitHub Repo stars](https://img.shields.io/github/stars/GuangSTrip/Awesome-Spatial-Intelligence) https://github.com/GuangSTrip/Awesome-Spatial-Intelligence
	- ![GitHub Repo stars](https://img.shields.io/github/stars/wufeim/awesome-3d-spatial-reasoning) https://github.com/wufeim/awesome-3d-spatial-reasoning
	- ![GitHub Repo stars](https://img.shields.io/github/stars/vulab-AI/Awesome-Spatial-VLMs) https://github.com/vulab-AI/Awesome-Spatial-VLMs
	- ### Neuroscience-Inspired
		- https://github.com/BioRAILab
-
## Survey
	- ==[[@Multimodal Spatial Reasoning in the Large Model Era: A Survey and Benchmarks]]==
	  collapsed:: true
		- [港科大（广州）等联合发布多模态空间推理综述：为大模型画下空间理解的未来蓝图](https://mp.weixin.qq.com/s/PFnC8DE2m9U2n8gWVD7GNw?scene=1&click_id=22)
		- ### 1) 这篇论文“讲了什么”（一句话概览）
		  collapsed:: true
			- 论文是一个**系统性综述 + 开源评测基准**：它把“多模态大模型（MLLM）在空间推理（spatial reasoning）”上的最新工作做了分类、比较、总结，并同时整理/提供了一套用于评估这类模型在二维/三维/视频/音频/具身任务上空间推理能力的公开基准与评测协议。作者目的既是把现状梳理清楚，也为后续研究者提供统一的评测工具。
		- ### 2) 为什么这事儿重要？（用生活化例子）
		  collapsed:: true
			- 人类能理解“杯子在桌子左边”“沙发后面有一扇窗”并据此做事（比如机器人去拿杯子）。这就是**空间推理**，它涉及相对位置、距离、角度、遮挡、视角变化、时间变化等。论文指出，要把这种能力赋给大模型，需要把图像/视频/点云/声音等多种感知通道和语言进行融合。MLLM（multimodal large language models）是目前最有希望的工具，但它们在空间理解上仍有明显短板，需要专门的研究和标准化评测。
			- 举例：给机器人一句话“把杯子从桌子右边移到盘子左边”，模型要理解什么是“右边/左边”、杯子/盘子相对位置，还要规划实际动作——这涉及从基础定位到规划执行的一系列空间能力。论文正是把这些能力类别化并讨论如何评测。
		- ### 3) 论文的主要贡献（作者自己怎么说）
		  collapsed:: true
			- 作者主要做了三件事：
				- **全面的分类/综述**：给出一个 taxonomy，把空间推理任务（2D/3D、导航、关系推理、生成等）、方法（测试时强化、后训练、结构改进、可解释性）和新模态（音频、第一视角视频）系统化整理。
				- **评测基准汇总与开放实现**：收集并介绍了大量近期 benchmarks（2D、3D、视频-文本、音频-视觉等），并发布了开源代码与评测套件以便标准化对比。论文提供了时间线和表格总结这些数据集与任务。
				- **分析与未来方向**：总结了当前 MLLM 在空间推理上的关键问题（例如表示不平衡、注意力侧重语义共现而非几何关系、数据标注困难等），并给出可行的研究路线与建议。
		- ### 4) 论文把“空间推理”分解成哪些子能力？（便于理解）
		  collapsed:: true
			- 论文列了十类能力，便于我们把问题拆开来看（并据此设计任务/基准）：
				- 定位（Localization）——在2D/3D里找物体；
				- 关系/几何（Relation & Geometry）——上/下/左/右、距离、角度；
				- 导航与规划（Navigation）——路径规划、最短路；
				- 模式/视角（Pattern & Perspective）——识别对称、不同视角下相同物体；
				- 缩放/变换（Scaling/Transformation）——尺寸与坐标变换保持关系；
				- 上下文（Contextualization）——同一位置在不同场景的含义（如房间 vs 飞船）；
				- 3D 生成（3D model generation）；
				- 场景建模（Environment modeling）；
				- 交互与实时感知（Sensing & Interaction）。
			- > 举例说明：在“定位”问题上，模型不只是回答“有杯子吗？”，而是回答“杯子相对于盘子是在左边还是右边，离盘子多远”。
		- ### 5) 作者如何把提升空间推理的研究方法归类？（三大方向，带举例）
		  collapsed:: true
			- 论文把方法分成几大类，我们逐一用直白例子说明：
				- **A. 测试时扩展（Test-time scaling / Prompting / Tool use） — 不改模型，改推理方式**
				  collapsed:: true
					- **Prompt engineering（提示设计）**：对模型下特殊格式的“指令/链式思考”提示，让模型把问题分步骤解决。论文观察到：纯文本的“长链式思考（CoT）”对空间问题效果有限，反而需要**结构性空间提示**（比如坐标、参考系、场景图）。例如：给模型一个“把房间鸟瞰图+坐标” 的提示，能显著提升“谁在谁左边”这类问题的准确率。
					- **工具使用（tool use）**：在推理时调用外部视觉模块（检测/深度估计/BEV鸟瞰图/视角合成）并把这些输出作为证据喂给模型。好处是不用重训就能引入几何信息；问题是工具错误会传播且输出缺统一格式。举例：先用目标检测画框、再把框信息以结构化 tokens 传给 LLM。
					- **多路径与检索（self-consistency / RAG / tree-search）**：在推理时多抽样、多路径选择共识，或检索地图/知识库来减少幻觉（hallucination）。例如 VISUOTHINK 做的是先在“视觉—文本空间”做树搜索再选择最佳解。
					- > 小结（类比）：这相当于不换司机或车，而是给司机更好的导航提示、额外的雷达数据和多条备选路线再投票——能在不改车的情况下跑得更稳。
				- **B. 后训练（Post-Training：SFT / RL）——在模型之上做进一步训练**
				  collapsed:: true
					- **监督微调（SFT）**：用空间标注数据把模型调到能直接判断几何关系。比如把“左/右/前/后”与坐标绑定的训练样本喂进去，让模型学到把坐标信息当作事实来利用。论文提到有从静态到时序（video/egocentric）的数据集来训练，效果随数据与课程（curriculum）设计显著提升。
					- **强化学习（RL）**：在需要连续决策或长时奖励的任务（例如导航、行动规划）用 RL 训练。优秀做法包括设计**过程级奖励**（不是只看最终成功，而是中间每一步是否几何正确），以及自我对弈/自玩（self-play）来产生更多训练信号。论文指出 RL 可以改善动态一致性与长期规划，但计算代价高、奖励稀疏是主要难点。
				- **C. 架构改进（Model design / Spatial modules）——直接改变模型结构**
				  collapsed:: true
					- 两类主流手段：
						- **输入增强（Input-level augmentation）**：把深度图、坐标 token、标记通道或多视角图像直接作为输入，告诉模型“这是几何信息”。例如把 (x, y) 坐标当成文本 token 附加在问题里，或把单张图像并行喂入RGB+marker通道。好处是侵入性小、实现简单，但依赖检测/深度预测质量。
						- **专门的空间模块（Spatial encoders / fusion blocks）**：在视觉侧或 LLM-连接层加入专门保留几何结构的网络（例如空间增强融合块、3D-aware encoder、relation-aware attention 等），这些模块保持多尺度的拓扑信息进入 LLM。论文列举了多种这类设计并报告显著提升（但复杂度也上升）。
		- ### 6) 可解释性方面：为什么 MLLM 空间能力不够？（诊断 + 解决方向）
		  collapsed:: true
			- 论文汇总了一系列可解释性研究，关键结论包括：
				- **表示不平衡（representation imbalance）**：视觉嵌入往往比位置/几何编码“更强”，导致模型依赖语义共现（objects tend to co-occur）而不是实际几何关系（attention 聚焦在物体而非物体间关系）。研究者提出通过规范化视觉token幅值、注入中间层几何特征来恢复空间敏感性。
				- **注意力偏差（attention bias）**：大模型的注意力多数权重指向单个对象区域，而不是关注对象-对象之间的相互关系。解决方向包括关系感知的 self-attention 设计（例如 RelatiViT）或在推理时动态调整注意力置信度。
				- **缩放报酬递减（scale diminishing returns）**：单靠增加数据与模型规模并不能彻底解决空间问题，空间能力更依赖于**几何化的视觉前端**与跨视图融合机制。简言之：更多数据不等于更好的几何理解。
		- ### 7) 3D 空间推理的特殊性（3D grounding / scene QA / 3D generation）
		  collapsed:: true
			- 论文把 3D 部分分得很细，内容要点如下：
			- **3D Visual Grounding（给一段语言在 3D 场景中定位目标）**
				- 三种输入策略：
					- **直接用 3D 表示（点云/体素）**：把点云等直接编码到模型里（更真实的3D信息，但结构复杂、数据稀缺）。
					- **多视角 2D 输入**：通过多张视角图利用现有2D MLLM 能力（把问题转换为“多张图+拼接提示”），缺点是视角不一致需要对齐策略。
					- **2D+3D 混合**：结合深度估计或部分3D特征（hybrid），通常在性能和复杂度上取得折中。
			- **3D Scene Reasoning & QA**
				- **训练-required 方法**：把 3D 特征（场景图、BEV、3DGS）对齐到 LLM，通过投影层或 Q-Former 等模块 fine-tune 模型以理解 3D 场景。
				- **训练-free 方法**：用 progress prompting、视角集合策略、Think-Program-Rectify 循环等利用“冻结的” MLLM 做 3D 推理（优点少训练成本，但推理复杂且受限于原始模型能力）。
			- **3D 生成（layout / programmatic generation）**
				- 两种思路：直接让 LLM 输出布局位置（可能导致不合物理的重叠），或让 LLM 生成**程序/脚本**（Blender 脚本、CAD 语句等），后者更可控、更适合精确几何建模。论文展示了用 LLM 生成 CAD/Blender 脚本的案例（把空间推理转为“可执行程序”）。
		- ### 8) 具身（embodied）任务中的空间推理：VLN / VLA / EQA 等
		  collapsed:: true
			- 论文讨论了把 MLLM 用作具身代理（robot/agent）大脑时面临的挑战与方法要点：
				- **输入要更空间化**：加入深度、点云、全景图等对导航与动作规划很关键。
				- **多任务预训练/联合训练**：把感知、推理、动作三条线一起训练能提升实用性。
				- **显式的中间状态**：把“意图→可行解→动作”拆成子步骤，用 affordance 预测或 goal-state 作为中间表征对长期任务有利。
			- > 举例：视觉—语言导航（VLN），一句话“带我到厨房的水槽附近”需要理解“厨房是什么样子、如何从当前位置到达、何谓‘水槽附近’的可接受半径”，这就需要长期记忆、地图库与实时视角转换。论文指出 MLLM 在这种跨时空任务上仍需更多专门设计。
		- ### 9) 论文对 Benchmarks（评测套件）的整理（非常详尽，且是论文的重要付出）
		  collapsed:: true
			- 论文把 benchmark 分类并列出代表性数据集（并给出时间线）：
				- **2D Image-Text Benchmarks**：早期如 Visual Genome、SpatialSense，到最近的 SPACE、BLINK、DriveMLLM 等，覆盖从基础相对位置到复杂场景。
				- **3D Benchmarks**：例如 SAT、SpatialRGPT-Bench、Spatial457 等，专注 3D 定位、6D 推理与多对象关系。
				- **Video-Text / Spatio-temporal Benchmarks**：VIS-100K、SPACER-151K、ST-ALIGN、EGO-ST、Video-R1 等，强调时间一致性、路径/事件定位与时序推理。
				- **其他模态**：Audio-Visual（如 SpatialSoundQA、STARSS23）、CAD/程序生成（CAD-GPT）和模拟任务（MM-ESCAPE）等也被纳进来以扩展评测覆盖。
			- 论文不仅列数据集，还对常见评测指标、标注难点、合成方法与现实验证做了深入讨论，并指出当前 benchmark 的短板（例如标注成本高、模板化数据导致泛化差、视频与音频标注稀缺）。
		- ### 10) 论文总结的“主要结论”（核心要点，便于记忆）
		  collapsed:: true
			- 我把论文的核心结论归纳成几条，便于你把握重点：
				- **MLLM 在空间推理上有潜力，但现状并不成熟**：它们在语义理解上强，但在几何/关系推理上仍弱（常依赖共现而非几何事实）。单纯放大模型或数据并不能彻底解决问题。
				- **方法层面的有效组合是关键**：测试时的工具集合（检测、深度、BEV）+结构化提示、以及后训练（SFT/RL）三者联合使用，通常能带来较大提升。
				- **输入端的几何证据非常重要**：无论是深度图、坐标 token，还是多视角/点云证据，都会显著提高模型的 3D 感知与定位能力。
				- **可解释性研究表明问题的根源在表征与注意力机制上**：需要把“关系”而非单个对象作为一等公民去建模（relation-aware attention / scene graph / 3D-aware modules）。
				- **评测基准仍不完备**：虽然涌现很多新 benchmark，但仍存在标注成本、模态覆盖不均、模板化与泛化不足的问题，建议算法-数据协同设计（algorithm-data co-design）。
		- ### 11) 论文列出的研究建议 / 未来方向（可作为你后续工作的灵感）
		  collapsed:: true
			- 论文给出一些明确可操作的研究方向，包括但不限于：
				- 构建**持久的对象中心场景记忆（object-centric scene memory）**，以跨视角/跨时刻维持物体状态；
				- 标准化工具输出 schema（统一坐标系、置信度），以便把多工具结果可靠地融合；
				- 发展成本感知（budget-aware）的控制器，在计划执行中动态选择 Plan–Execute vs ReAct 模式；
				- 加强音频与第一视角（egocentric）空间推理研究，因为这两类模态更贴近机器人/助理的真实工作场景；
				- 推进可程序化的 3D 生成（把空间推理输出为可执行的 Blender/CAD 脚本），提升可控性。
		- ### 12) 用一个完整的“通俗例子”把整个流程串起来（帮助小白理解）
		  collapsed:: true
			- 假设我们要让一个家用机器人完成任务：“把客厅桌子上红色杯子放到餐桌左边。” 所需步骤与论文中对应的研究点：
				- **感知**：机器人用相机采图（可能是多视角），用深度估计/点云恢复几何（论文推荐输入增强）。
				- **解析指令**：把“红色杯子”“餐桌左边”解析成语义+坐标 anchor（提示工程里建议把坐标/参考系加入 prompt）。
				- **定位与关系推理**：模型判断“杯子在哪里，相对餐桌是什么方向、距离”，需要关系-aware attention 或外部工具提供准确坐标。
				- **规划动作**：在具有物理可行性检查的环境下规划移动路径（强化学习可提供过程级奖励以保证每一步都合理）。
				- **执行并复核**：执行过程中持续用视觉/深度检查，若偏差则用 Think-Program-Rectify 等循环自纠。
			- 论文的主张是：构建一个包含**几何输入 + 结构化提示 + 专门空间模块 + 训练/微调/必要时 RL 优化**的混合体系，比单独依赖 LLM 更可靠。
			  
			  ---
		- ### 13) 如果你要马上做后续工作（几条实操建议）
		  collapsed:: true
			- **从 benchmark 入手**：选择论文中提到的一个中等难度的数据集（如 VIS-100K 或 SpatialRGPT-Bench / SAT）进行复现与基线测试。论文和开源仓库有整合列表与实现。
			- **先做输入增强**：在现成 MLLM 基础上先加深度或坐标 token，看最小代价能带来多少提升。
			- **做工具链集成**：把检测/深度/BEV 的输出结构化为统一 schema（位置、置信度、参考系），以便下游 LLM 使用。论文建议标准化 tool 输出有助于复现性。
			- **设计可解释性 probe**：分析注意力/embedding 幅值，验证模型是否真正使用了几何信息（论文给了可解释性研究方向）。
		- ### 14) 结尾小结（一句话回顾）
		  collapsed:: true
			- 这篇论文把“多模态大模型的空间推理”做了全面的地图式整理：**它既告诉你目前哪些方法有用（提示、工具、后训练、结构模块），也指出为什么仍不够（表示与注意力偏差、数据与标注问题），并提供了大量 benchmark 与实践建议，便于科研或工程应用的下一步落地**。如果你要做相关研究，这篇 survey + 其开源仓库是非常值得作为起点的参考资料。
	- [[@A Survey of Large Language Model-Powered Spatial Intelligence Across Scales: Advances in Embodied Agents, Smart Cities, and Earth Science]]
	- [[@Aligning Cyber Space with Physical World: A Comprehensive Survey on Embodied AI]]
	- ~~trivial~~
	  collapsed:: true
		- A review of embodied intelligence systems: a three-layer framework integrating multimodal perception, world modeling, and structured strategies
		- **《Embodied Intelligence: A Synergy of Morphology, Action, Perception and Learning》** ([Liu et al., 2025](https://dl.acm.org/doi/abs/10.1145/3717059))
		  collapsed:: true
			- 该文发表于 *ACM Computing Surveys*，系统性梳理了具身智能中形态、动作、感知与学习的相互作用关系，强调了主动探索（Active Perception）与空间语义理解的融合，为3D场景理解提供了认知框架。
			  
			  🔍 *亮点*: 提出“形体-环境-认知耦合”理论，强调通过环境交互提升空间推理与记忆模块性能。
		- **《A Survey of Large Language Model-Powered Spatial Intelligence Across Scales》** ()
		  collapsed:: true
			- 探讨了如何利用大语言模型（LLMs）增强空间智能，包括具身代理（Embodied Agents）、城市级理解与地球尺度建模。
			  
			  🔍 *亮点*: 对“语义驱动探索（Semantic-driven Exploration）”进行了综述，提出结合语言与神经空间记忆模型的新方向。
		- **《Semantic Mapping in Indoor Embodied AI – A Comprehensive Survey and Future Directions》** ([Raychaudhuri & Chang, 2025](https://www.i-newcar.com/uploads/ueditor/20250123/2-250123113G2J3.pdf))
		  collapsed:: true
			- 全面总结了室内具身AI的语义地图构建方法，从SLAM到语义SLAM，再到基于Transformer与3D视觉的语义建图。
			  
			  🔍 *亮点*: 强调主动探索与语义引导路径规划的融合，是理解“Move to Understand a 3D Scene”思想的直接延伸。
		- When LLMs step into the 3D World: A Survey and Meta-Analysis of 3D Tasks via Multi-modal Large Language Models
		- Scene Representations for Robotic Spatial Perception
		- Spatial Navigation and Memory: A Review of the Similarities and Differences Relevant to Brain Models and Age
		-
		- | 论文 | 核心内容简介 | 与你方向的相关性 |
		  | ---- | ---- | ---- |
		  | Vision‑Language Navigation with Embodied Intelligence: A Survey (2024) [arXiv](https://arxiv.org/abs/2402.14304?utm_source=chatgpt.com) | 专注于视觉-语言导航 (VLN)，回顾了“具身智能”(embodied intelligence)环境下视觉＋语言导航的方法、数据集、挑战。 | 虽然重点在语言＋视觉，而非3D主动探索，但对“具身代理如何在环境中导航”整体非常有参考价值。 |
		  | A Survey on Vision‑Language‑Action Models for Embodied AI (2024) [arXiv](https://arxiv.org/abs/2405.14093?utm_source=chatgpt.com) | 专注于 “视觉-语言-动作”(VLA) 模型，即感知＋语言＋动作融合的代理系统。涵盖体系结构、训练范式、实际任务。 | 在你探讨“主动探索”时，VLA 是一个重要维度：不仅“理解”环境、还“行动”于环境。 |
		  | Embodied Navigation (2025) [link.springer.com+1](https://link.springer.com/content/pdf/10.1007/s11432-024-4303-8.pdf?utm_source=chatgpt.com) | 一个系统综述，覆盖具身导航 (embodied navigation) 的感知、导航、效率优化、实环境部署挑战。 | 与 3D 空间理解＋主动探索高度相关，是你理解“导航”体系结构＋挑战的好入口。 |
		  | Semantic Mapping in Indoor Embodied AI ‑ A Comprehensive Survey and Future Directions (2025) [arXiv+1](https://arxiv.org/html/2501.05750v1?utm_source=chatgpt.com) | 专注于室内具身 AI 中 **语义地图构建** 方法：地图结构、编码形式、任务驱动表示。 | 地图/空间记忆模块与“空间理解＋记忆”密切相关，非常贴合你模型改进的记忆模块方向。 |
		  | Semantically‑aware Neural Radiance Fields for Visual Scene Understanding: A Comprehensive Review (2024) [arXiv](https://arxiv.org/abs/2402.11141?utm_source=chatgpt.com) | 虽不是专门导航，但回顾了 NeRF (神经辐射场) 在 3D 场景语义理解中的进展：语义分割、3D 物体检测、场景补全。 | 如果你模型涉及 3D 场景重建 ／ 语义理解（而不仅导航）部分，这篇综述很有价值。 |
## Neuroscience-Inspired
	- ==[[@From reactive to cognitive: brain-inspired spatial intelligence for embodied agents]]==
	  collapsed:: true
		- ## 一、核心问题与动机（一句话总结）
			- 当前很多基于大模型或端到端强化学习的“具身智能”系统都是**反应式（reactive）**的——只根据当前视觉/文本输入做决策，不把环境的空间知识长期保存为可以复用、推理的内部模型。
			- 生物（人脑）通过三类互补的空间记忆（**地标 landmarks、路线 route、全局地图/认知地图 survey**）来做高效、灵活的导航与推理。
			- 论文的目标是把这个“脑启发”的空间记忆机制，构造成一个工程系统（BSC-Nav），用在带传感器和执行器的机器人上，从而提升长期导航、开放词汇目标、指令分解与问题答复等能力。
		- ## 二、总体架构（高层直观类比）
			- 把论文的架构想成人的“旅行日记 + 地图 + 临时记忆”三部分：
				- **地标记忆（Landmark memory）** —— 像你把重要位置（“厨房的咖啡机”、“客厅靠窗的沙发”）写成条目，包含位置、类别、和一句描述（由大模型生成），用于快速定位和推理。
				- **认知地图 / 体素化认知图（Cognitive map / voxelized map）** —— 把机器人走过的轨迹与视觉补丁编码成三维网格（体素），在每个体素里缓存若干视觉特征（buffer），长期保存多视角信息，支持从任意视角检索与比对（类似一张可查询的鸟瞰地图）。
				- **工作记忆（Working memory）** —— 当任务来了（文本目标或图片目标），系统根据目标复杂度在“地标”和“认知地图”之间智能检索、合成候选位置，再按“置信度+距离”排序产生探索序列，最终用低级规划去执行并用大模型（MLLM）做目标验证与动作建议。
				- > 举例：如果指令是“去找蓝色的经典茶壶放在茶盘上”，系统会：
				  1. 在**地标记忆**里先检索“厨房／茶盘／茶具”等快捷线索（快速候选）；
				  2. 若不够精细，使用大模型把文字扩充成“想象图”（text→image），用想象图编码特征去匹配**认知地图**中的体素特征，得到更精确的候选位置；
				  3. 对候选位置按“相似度（置信）”和“距离”打分，规划一个高效的探查顺序，执行并在到达后让 MLLM 结合视图做最终确认。
		- ## 三、关键实现要点（方法细节，便于复现）
			- ### 1) 观测与坐标变换
				- 每个时间步观测包含 RGB 图像、深度图与位姿 (X,Y,yaw)。把图像 patch 的中心像素用深度反投影到相机坐标，再通过固定的相机→机器人基座、以及基座→世界变换，得到该 patch 在世界坐标系下的 3D 点，最后用体素网格（voxel）索引存储特征。公式和坐标变换细节论文中给出。
			- ### 2) 地标记忆（Landmark memory）
				- 一个地标条目是四元组：位置 θ=(X,Y,Z)、类别 c、置信 ρ、以及由 GPT-4o 生成的文本描述 T（包含材质、形状、语境）。
				- 检测器采用 open-vocabulary 检测（论文用 YOLO-World），并有置信阈值与空间重叠融合机制，防止重复记忆（靠邻近距离与类别判断并融合）。
			- ### 3) 认知地图（Cognitive map / voxel buffer）
				- 地图由三维体素网格 Mcog 表示：每个体素 v 保存一个容量 B 的特征缓冲区（多视角特征）。视觉编码用 DINO-v2 抽取 patch-level 特征。观测的 patch→体素映射按投影与网格离散化完成。
				- **惊讶驱动（surprise-driven）更新策略**：对新进来的特征 f_new，计算其与周围 n-hop 体素缓冲区中已有特征的平均距离（即“surprise”分数）；只有当超过阈值 τ（默认 0.5）时才加入缓冲，以避免重复存储稳定/冗余的特征，并在满缓冲时替换“最不惊讶”的特征。这个思路借鉴了神经科学中的自由能/预测误差最小化原理。优点是：保持多视角与多样性，同时控制记忆容量。
			- ### 4) 工作记忆的分层检索（检索策略）
				- **先试地标记忆（快速路径）**：对于简单、类别级目标（“沙发”），直接在地标记忆里用 MLLM 做语义匹配并返回候选坐标。因为地标是稀疏且抽象的，检索很快。
				- **复杂目标→认知地图（精确路径）**：把文本目标用 MLLM 扩充成细节，再用文本→图像（Stable Diffusion）“想象”目标外观，编码为视觉特征，去匹配认知地图的体素缓冲特征，得到更细粒度候选地块（通过相似度计算 + DBSCAN 聚类返回K个候选）。这相当于“先在脑里想象要找的样子，再在记忆的地图里搜”。
			- ### 5) 候选排序与探查序列
				- 对每个候选位置计算综合优先级： `H_i = λ·p_i + (1−λ)·(1 − d_i/d_max)` ，其中 p_i 是候选存在概率（置信度或相似度），d_i 是与当前点的欧式距离，λ 默认 0.5。用这个分数决定先去哪个候选点，从而既考虑“最可能在那儿”，也考虑“离得近”。
			- ### 6) 低级规划与到达验证
				- 低级运动在仿真中用 Habitat 提供的贪心最短路径；在真机上用 A* 作全局规划、TEB 做局部避障（输出速度控制）。到达候选点后做 360° 扫描，用 CLIP 等把图像嵌入与目标文本/图像比对，再把最匹配的图像交给 GPT-4o 做最终“是否到达”的判定和姿态微调建议（affordance）。
		- ## 四、实验设计与关键结果（要点与数字）
		  collapsed:: true
			- 论文在仿真和真机两个层面做了大量评估，任务覆盖从基础导航到更高阶的指令理解与问答。
			- ### 仿真实验（Habitat，MP3D / HM3D）
				- **基础导航任务**：包括 Object-Goal Navigation (OGN)、Open-Vocabulary Object Navigation (OVON)、Text-Instance Navigation (TIN)、Image-Instance Navigation (IIN)。在 62 个室内场景、8195 次实验里，BSC-Nav 在**类别级任务**上 SR 达到：HM3D 78.5%、MP3D 56.5%，比最好的基线（UniGoal 等）提高显著（例如超过 15–24%）。在 OVON（79 类）中也能零样本取得 ~40% 左右的 SR，超过一些有监督方法。IIN（图像级）SR 可达 71.4%。同时 SPL（效率）也显著优于基线，说明不仅能找到目标，路径也更短更高效。
			- ### 高阶任务
				- **长航程指令导航（VLN-CE R2R）**：在 1000 条人类长指令的零样本测试上，BSC-Nav 达到 **38.5% SR**、**53.1% SPL**，SR 比最强监督方法低约 8.5%，但 SPL（效率）显著领先，表明结构化记忆有助于高效完成复杂指令。
				- **主动式具身问答（A-EQA，184 问）**：使用 LLM-Match 评估（GPT 对答案打分），BSC-Nav 得分 **54.6%**，显著优于盲模型、被动探索等基线，尤其在需要细粒度定位的题型（定位、对象状态识别）上提升明显，但仍落后于人类（论文指出差距约 27.5%）。
			- ### 真机部署（室内两层 ~200 m²）
				- 在真实机器人上对 OGN、TIN、IIN 做了 75 次试验（15 个目标，每个目标 5 次起点随机）：
				- 多数目标至少 3/5 次成功，IIN 在 4/5 个目标上达成 100% 成功率；
				- 平均路径长度 ~23.4 m，平均速度 ~0.76 m/s，最终到达距离多数 < 2.5 m，表现实用且稳定。论文还展示了移动操作（grasp、pour 等）与导航结合完成的“做早餐”多步任务演示。
		- ## 五、作者的结论、科学与工程意义
			- **科学意义**：论文把认知科学中“地标／路线／survey”三类空间知识具体化、工程序列化，并展示了它们如何与现代视觉编码与大模型（如 DINO-v2、GPT 系列）联动，从而实现更类似“认知”而非单纯“反应”的具身智能系统。论文还用了“惊讶驱动更新”把自由能（free-energy）思想引入工程实现，体现了神经学原理到算法的映射。
			- **工程意义**：结构化空间记忆提高了**泛化（zero-shot）能力**、导航**效率**和多模态任务表现，且能从仿真较好迁移到真机，这是当前很多端到端方法难以做到的。论文代码和部署细节开源，利于复现。
		- ## 六、局限与未来方向（论文中也指出）
			- 论文诚实地指出若干局限与后续改进方向，包括：
				- **动态与非结构化场景扩展**：当前方法在静态/室内场景表现好，拓展到高度动态或户外场景还需改进。
				- **内存效率与实时性**：体素缓冲和想象检索需要计算资源，真机或大规模部署时需更高效的内存压缩/检索策略。
				- **更全面的“具身智力测评”**：作者提出构想类似“具身图灵测试”的评估框架来衡量空间认知，未来可设计包含环境变化、协同任务、多步叙述等更严格的基准。
		- ## 七、用通俗小白也能懂的总结（3 分钟速记版）
			- 问题：现在很多机器人只“看到就反应”，不会把空间记下来复用。
			- 灵感：生物把空间分成**地标（记点）→ 路线（记怎么走）→ 地图（记整体）**三份知识。
			- 方法：论文做了一个系统 BSC-Nav，把视觉特征和大模型结合，做到（1）记重要点（地标）；（2）把走过的视觉片段放进三维网格（认知地图）并用“惊讶”规则只存新东西；（3）遇到任务时智能检索、想象（text→image）并匹配地图，按“可能性+距离”去搜，最后用大模型确认。
			- 效果：在仿真和真实机器人上，这套方法在多种导航和问答任务上都明显好于没有“长期空间记忆”的方法，既更会找东西，也更会节省路程。
	- [[@Mind Meets Space: Rethinking Agentic Spatial Intelligence from a Neuroscience-inspired Perspective]]
	- [[@Neural Brain: A Neuroscience-inspired Framework for Embodied Agents]]
	- [[@RoboMemory: A Brain-inspired Multi-memory Agentic Framework for Interactive Environmental Learning in Physical Embodied Systems]]
	- ~~trivial~~
	  collapsed:: true
		- Neural Brain: A Neuroscience-inspired Framework for Embodied Agents
		- Towards Neuro-Inspired Reasoning of AI Agents
		- A review of neuroscience-inspired machine learning
		- Brain-inspired learning, perception, and cognition: A comprehensive review
		- A review of neuroscience-inspired frameworks for machine consciousness
		- Personalized artificial general intelligence (agi) via neuroscience-inspired continuous learning systems
		- Synergizing Neuroscience and Artificial Intelligence: Brain-Inspired Architectures for Enhanced Performance and Neural Computation Insights
		- Brain-inspired artificial intelligence: A comprehensive review
		- **《Neural Brain: A Neuroscience-Inspired Framework for Embodied Agents》** ()
			- 将神经科学中关于**海马体、位置细胞、网格细胞与记忆回放**的发现引入具身智能模型，提出了“Neural Brain”框架。
			  
			  🔍 *关联*: 非常适合你提到的研究方向——通过模拟人脑的空间记忆机制改进空间表示与探索策略。
		- **《Mind Meets Space: Rethinking Agentic Spatial Intelligence from a Neuroscience-inspired Perspective》** ()
			- 从认知科学与神经计算角度，探讨具身智能系统如何通过“神经编码结构”形成空间地图。
			  
			  🔍 *亮点*: 将神经回放与经验压缩（Experience Replay）机制引入空间导航任务，为主动探索提供神经层面解释。
## Competitors
	- ==[[@Move to Understand a 3D Scene: Bridging Visual Grounding and Exploration for Efficient and Versatile Embodied Navigation]]==
	- [[@3D-Mem: 3D Scene Memory for Embodied Exploration and Reasoning]]
	- [[@OmniNav: A Unified Framework for Prospective Exploration and Visual-Language Navigation]]
	- [[@OpenFrontier: General Navigation with Visual-Language Grounded Frontiers]]
	- ~~trivial~~
	  collapsed:: true
		- **《Embodied Intelligence for 3D Understanding: A Survey on 3D Scene Question Answering》** ()
		  collapsed:: true
			- 聚焦于3D场景理解与问答（3D Scene QA），强调**探索与语义推理的联动**。
			  
			  🔍 *相关性*: 在“Move to Understand a 3D Scene”中提出的“从视觉导航到语义理解”桥梁思想，与本综述中“主动查询式探索”高度契合。
		- **《Physical Scene Understanding》** ()
		  collapsed:: true
			- 该篇发表于 *AI Magazine (AAAI)*，从认知科学角度探讨了物理场景理解、物体交互与主动感知的统一框架。
			  
			  🔍 *亮点*: 强调“预测性建模（Predictive Modeling）”与“具身交互”的结合，为构建空间智能的认知先验提供理论支持。
		- **《Scene Representations for Robotic Spatial Perception》** ([Mascaro & Chli, 2024](https://www.annualreviews.org/content/journals/10.1146/annurev-control-040423-030709))
		  collapsed:: true
			- *Annual Review of Control, Robotics, and Autonomous Systems* 的顶级综述，系统总结了机器人空间感知的场景表示学习方法。
			  
			  🔍 *关联*: 对比现有的空间表示（occupancy maps, voxel grids, neural fields），并提出改进空间记忆模块的前景。
		- collapsed:: true
		  
		  BIG: a framework integrating brain‑inspired geometry cell for long‑range autonomous exploration and navigation (2025) [SpringerOpen](https://satellite-navigation.springeropen.com/articles/10.1186/s43020-024-00156-3?utm_source=chatgpt.com)
			- 提出了「几何细胞 (geometry cell)」「头向细胞 (head-direction cell)」等脑启发模块，用于探索＋导航任务。
			- 虽可能不在 顶会 （但为2025年），直接体现神经科学机制在导航任务中的应用。
			- 与你“从神经机制启发设计”方向吻合。
		- collapsed:: true
		  
		  Mapping High‑level Semantic Regions in Indoor Environments without Object Recognition (2024) [arXiv](https://arxiv.org/abs/2403.07076?utm_source=chatgpt.com)
			- 虽不是完全“探索”任务，但关注室内环境中高层语义区域（如房间、区域类型）映射，结合 embodied 导航。
			- 对理解“3D 空间理解＋语义区域”有帮助，尤其有利于模型理解房间／区域结构这一维度。
		- collapsed:: true
		  
		  Deep Learning‑Emerged Grid Cells‑Based Bio‑Inspired Navigation (2025) [MDPI](https://www.mdpi.com/1424-8220/25/5/1576?utm_source=chatgpt.com)
			- 虽更多偏向机器人/导航而非 3D 场景地理解，但从神经科学–网格细胞视角探讨了深度学习模型如何产生“格网”状的空间编码。
			- 非常适合你“借鉴神经机制改进 3D 空间理解模型”这一方向。