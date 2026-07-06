---
layout: post
title: "从 Transformer、SSM 到 Object-Centric World Model：压缩、记忆与状态建模"
date: 2026-07-06 18:00:00 +0800
categories: [ai, world-models]
---

> "压缩即智能"是一个流行的口号。如果它成立，那么把历史激进压缩成固定状态的 RNN/SSM 似乎应该比"什么都存着"的 Transformer 更聪明——但现实恰恰相反。问题出在哪里？本文从 Transformer 与 RNN/SSM 对"历史"的不同处理方式出发，论证真正的关键不是压缩多少，而是**压缩的时机**；由此推出混合记忆系统的一般设计原则；再把这套原则从语言模型迁移到视觉世界模型，结合 JEPA 与 Dreamer/RSSM 的思想，给出一个 object-centric world model 的完整架构方案——包括四层对象记忆、交互建模、跨时间身份绑定、训练目标，以及一个可以直接上手的最小可行版本。

---

## 1. 核心问题：Transformer 与 RNN/SSM 的本质区别

### 1.1 同一个目标，两种历史观

对于自回归预测，所有序列模型都在建模同一个分布：

$$p(x_{t+1} \mid x_{\le t})$$

差异不在目标，而在于它们如何**表示"历史" $x_{\le t}$**。这个看似实现层面的选择，实际上决定了两类架构的根本性格。

### 1.2 Transformer：显式保存历史，按需读取

Transformer 把历史 token 的表示原样保留为一个不断增长的集合（KV cache）：

$$M_t = \{(k_i, v_i)\}_{i=1}^{t}$$

预测时，当前 query 通过 attention 对这个集合做内容寻址的动态读取：

$$c_t = \sum_{i \le t} \alpha_{ti}\, v_i$$

注意这里发生了什么：历史本身**从未被合并或压缩**，每个位置的信息都独立存在、随时可访问；被"压缩"的只是每次读取时临时构造的摘要 $c_t$，而且这个摘要是**为当前 query 量身定制**的——下一个 token 的 query 不同，摘要就重新构造一次。因此 Transformer 的核心是：

> **历史作为可寻址外部记忆存在，模型在预测时根据当前 query 动态构造所需摘要。**

### 1.3 RNN / SSM：递归压缩历史，维护固定状态

RNN 和 SSM 走的是另一条路：把过去持续压缩进一个**固定维度**的状态：

$$s_t = f(s_{t-1},\, x_t)$$

预测完全基于这个状态：

$$p(x_{t+1} \mid s_t)$$

这条路的代价隐藏在时序里：状态容量固定，每来一个新 token 就必须做一次不可逆的取舍——哪些旧信息保留、哪些被新信息挤出。而做这个决定的时刻，**未来会问什么问题还完全未知**。因此其核心是：

> **模型必须在未来需求尚不确定时，就决定哪些历史信息写入状态、哪些被遗忘。**

### 1.4 最凝练的区别

$$\boxed{\text{Transformer：学习如何读取历史}}$$

$$\boxed{\text{RNN/SSM：学习如何把历史变成状态}}$$

一个学的是读取策略（历史保持原样），一个学的是写入策略（历史被就地改写）。后面所有的讨论，都是这个区别在不同场景下的展开。

而"把历史就地改写成固定状态"，正是**压缩**的另一种说法。于是一个诱人的推论浮现出来：既然流行的口号说"压缩即智能"，那么压缩得更彻底的 RNN/SSM，岂不是应该更聪明？下一节就从拆穿这个推论开始。

## 2. 为什么"压缩即智能"推不出 RNN/SSM 更强

### 2.1 口号哪里出了问题

"压缩即智能"背后的直觉是正确的：智能体必须从冗余的经验中提炼可复用的规律。RNN/SSM 确实更显式地执行历史压缩——但推不出它们应该更强，因为它们的压缩方式有四个结构性缺陷：

- **压缩得过早**：在信息刚到达、其未来用途还完全未知时就必须压缩；
- **容量固定**：无论输入多复杂，状态维度不变，信息必然发生挤压和覆盖；
- **不可逆**：一旦被挤出状态，原始信息永久丢失，没有恢复路径；
- **盲目选择**：整个取舍过程在不知道未来 query 的情况下进行，只能押注于"平均而言什么重要"。

### 2.2 Transformer 并非不压缩，而是推迟压缩

一个常见的误解是把 Transformer 描述成"不压缩、靠堆内存取胜"。更准确的说法是：Transformer 把压缩**推迟到 query 出现之后**：

$$c_t = \operatorname{Summary}(x_{\le t};\; q_t)$$

这个延迟带来了四个决定性的性质：

- **query-dependent**：摘要针对当前需求定制，而不是提前押注；
- **可重复执行**：每一层都可以对同一段历史做一次新的摘要，逐层深化；
- **多视角并行**：不同 attention head 可以同时采用不同的摘要视角（一个 head 看句法、一个看指代、一个看数值）；
- **一份历史，多种读法**：不同的未来 token 可以对同一段历史做完全不同的读取——历史只存一份，解读无限多种。

因此 Transformer 的优势不是"不压缩"，而是：

> **先保留细节，再按当前任务进行动态压缩。**

### 2.3 压缩发生在哪里：参数与上下文的分工

把镜头拉远，LLM 中的"压缩"其实至少发生在两个层面，且分工明确：

1. **参数压缩**：训练语料中反复出现的统计规律、世界知识和算法模式，被永久压缩写入模型参数 $\theta$——这是真正意义上的"提炼可复用规律"；
2. **上下文计算**：当前样本中的临时信息（这段对话、这份文档特有的内容）保留在上下文中，推理时按需动态组合——这些信息不该被"提炼"，因为它们不可复用、只能精确保留。

所以对口号更准确的修正是：

> **智能来自对可复用预测结构的压缩，而不是最激进的状态压缩。**

该压缩的（规律）狠狠压进参数；不该压缩的（实例细节）原样留在上下文。RNN/SSM 的问题正在于它把两类信息塞进了同一个固定状态。

## 3. 自然语言的双重结构：为什么它偏爱延迟压缩

### 3.1 语言不是均匀的"高熵"信号

上一节的分工之所以有效，根源在于自然语言本身的信息结构。语言并不是单纯的高熵信号，而是两类性质相反的信息的混合：

$$\boxed{\text{高度可压缩的结构} + \text{局部难以压缩的精确细节}}$$

### 3.2 可压缩的部分

这类信息在语料中大量重复、可以从规律中重建：

- 语法结构；
- 语义关系的组合模式；
- 叙事和篇章结构；
- 世界知识；
- 推理模式；
- 程序与代码的惯用结构。

它们是"可复用预测结构"的典型代表，适合被压缩进模型参数——学一次，处处可用。

### 3.3 难以安全压缩的部分

另一类信息则完全相反——它们是任意的、局部的、不可从规律推断的：

- 人名；
- 数字；
- 日期；
- 代码中的变量名；
- 随机字符串（ID、哈希、密钥）；
- 用户在对话中的临时定义（"下文中把 A 记作 X"）；
- 特定文档中的特定事实。

这些内容有一个共同特点：**有损压缩即毁灭**。"大约叫张什么"的人名、"差不多是三千多"的数字，在下游任务中等于没有。它们只能精确保留，不能被摘要。

### 3.4 两种架构面对这个结构的表现

Transformer 的分工恰好与语言的双重结构对齐：

$$\text{共享规律} \to \theta \qquad \text{当前上下文细节} \to \text{KV memory} \qquad \text{预测时} \to \text{content-based retrieval}$$

规律走参数通道，细节走记忆通道，互不干扰。而 RNN/SSM 试图把抽象结构和精确实例细节写入**同一个固定状态**——两类信息在有限容量中竞争，结果往往是：语义大意保住了，精确细节（那个具体的数字、那个变量名）被挤掉了。这就是长上下文任务中递归模型典型的失败模式：**信息干扰与选择性遗忘**。

## 4. 根本困境：我们到底应该保留多少历史

### 4.1 问题的不可解性

现在把问题推到最一般的形式。任何记忆系统设计都绕不开这个困境：

> **在未来 query 尚未出现前，无法准确知道哪些历史信息以后会被使用。**

这不是工程能力问题，而是信息论上的原则性障碍：最优压缩依赖于对未来查询分布的知识，而这个分布在压缩时刻不可得。因此**不存在一个普适、无损、固定大小的最优 summary**。任何声称"把任意长历史无损压缩进固定状态"的方案，都在某个查询分布下必然失败。

### 4.2 两个极端的得失

既然最优解不存在,就只能权衡。先看两个极端。

**全量 KV cache（Transformer 默认）：**

优势：
- 不需要在写入时判断重要性——把判断推迟到读取时；
- 未来可以重新访问任意历史位置，对任何查询分布都稳健；
- 精确的数字、名字、代码原样保留；
- 不会因为一次错误的摘要而永久丢失信息。

代价：
- 内存随上下文长度线性增长；
- attention 的读取成本随历史长度持续增加。

**单一递归 summary（RNN/SSM 默认）：**

优势：
- 固定内存，与序列长度无关；
- 天然流式更新，适合无限长输入；
- 每步计算恒定，高效。

代价：
- 有损且不可逆；
- 无法适应未来的任意查询；
- 典型失败模式：保留语义大意，丢失精确细节。

### 4.3 出路：按信息性质分层，而不是二选一

正确的问题不是"存还是压"，而是**"哪些信息可以安全压缩，哪些信息必须保留恢复路径"**。不同信息在时间尺度、可压缩性、精确性要求上差异巨大，就应该用不同的策略对待。一个稳健的记忆系统因此是四种组件的组合：

$$M_t = \left(C_t,\; s_t,\; E_t,\; R\right)$$

其中：

- $C_t$：**近期完整记忆**——最近的 KV 或局部窗口，全分辨率保留，因为近期信息还没到可以安全压缩的时候；
- $s_t$：**长期递归语义状态**——远处历史的大意可以放心压缩，这正是递归状态擅长的；
- $E_t$：**稀疏关键事件**——少数特别重要的历史点（关键结论、转折）单独精确保留；
- $R$：**可回读的原始内容**——最后的安全网：当压缩出错或遇到意外查询时，还能回到原文。

$$\boxed{\text{近期完整记忆} + \text{长期压缩状态} + \text{稀疏关键事件} + \text{必要时回读原始内容}}$$

这个四元组是全文的核心模板。后面无论讨论 LLM 的分层 attention 还是世界模型的对象记忆，都是它在不同模态下的实例化。

## 5. 分层压缩 Attention：模板在语言模型上的实例

把第 4 节的模板落到 LLM 的 attention 机制上，得到三个分辨率递减的访问层。

### 5.1 局部高分辨率记忆

保存最近一段的完整信息，不做任何压缩。它服务于：

- 局部语法（当前句子的组织）；
- 短期依赖；
- 精确引用（刚刚出现的数字、名字）；
- 连续的推理动力学（上一步到下一步的衔接）。

对应模板中的 $C_t$——近期信息的用途还高度不确定，任何压缩都为时过早。

### 5.2 中尺度稀疏检索

对较远的历史先做**温和**压缩（保留可辨识性的索引级压缩），再由当前 query 从中选出少量真正相关的位置做精读：

$$\mathcal{I}(q) = \operatorname{TopK}_j\, \operatorname{Score}(q,\, \bar{k}_j)$$

它服务于：

- 稀疏的远程依赖（远处的某个定义、某条前提）；
- 关键证据的按需调取；
- 实体或代码变量的长距离引用。

设计要点在于"温和压缩 + 保留回读"：压缩只用于降低检索成本，命中之后仍然读取精细内容——这正是模板中 $R$（恢复路径）原则的体现。

### 5.3 全局低分辨率摘要

对整个历史做高比例压缩，但让结果**始终可见**（无需检索即可访问）：

$$x_{1:T} \;\to\; \bar{x}_{1:T/m}$$

一个关键设计决定：它**不是单一 global vector**，而是一串保持顺序的低分辨率 memory tokens。这个区别很重要，因为 token 序列形式保留了：

- 顺序信息；
- 粗粒度的篇章结构；
- 多个可区分的语义区块；
- 以及最关键的——query-dependent 读取能力：attention 仍然可以按需侧重摘要中的不同部分，而单一向量对所有 query 只能给出同一个答案。

### 5.4 三层合成

$$\boxed{\text{近期精细} + \text{远期稀疏精读} + \text{全局粗读}}$$

近处看得清、远处找得到、全局有概念——三层各司其职,合起来以次线性的成本逼近全量 KV 的功能覆盖。

### 5.5 一个现实对应：它就是 DeepSeek 式 sparse attention

值得点明的是，这三层不是纸面推演——它精确对应了已经落地的架构。DeepSeek V4 风格的 sparse attention 大致由三个访问通道组成：

- **recent tokens**：最近若干 token 的完整精细上下文——对应 5.1；
- **CSA**（压缩稀疏检索）：对较远历史先做温和压缩形成索引，再由当前 query 选出少量位置精读——对应 5.2；
- **HCA**（高压缩全局通道）：对整个历史高比例压缩，形成低成本、始终可见的全局视野——对应 5.3。

$$\text{fine memory} + \text{compressed retrieval index} + \text{coarse global summary}$$

换句话说，第 5 节这套模板正是 DeepSeek 式 sparse attention 的去名字化抽象版。

但这里必须精确区分它**回答了什么、没回答什么**。Sparse attention 回答的是：

> **当前 query 应该读取哪些历史位置？**

而一个完整的语义层级还要回答另一个问题：

> **这些 token 组合起来构成了什么高层变量？**

前者只是在原有 token 图上做**边的稀疏化和访问的分层**——节点仍然是原始 token（或其压缩副本）；后者则要求产生**新的节点、新的状态、新的特征基**，比如一个显式的"实体状态"或"事件表示"，它不再是任何单个 token 的副本：

$$\text{sparse attention} \approx \text{稀疏边} + \text{分层访问}$$

$$\text{语义层级} \approx \text{新节点} + \text{新状态} + \text{新特征基}$$

所以严格地说，本节这三层（以及 DeepSeek 式 sparse attention）层次化的是**访问方式**，而非**语义表示本身**——它只做到了"读取侧"的分层，"表示侧"仍然是隐式的。

记住这个缺口：后文为对象世界设计记忆时新增的 **object state 层**（第 9.3 节的 $\{h_t^i\}$——持续存在的实体、遮挡期间的信念），正是这里缺失的"表示侧"——它显式地造出了新节点，而不只是重新组织对旧 token 的访问。这也是文本记忆与对象记忆最本质的一处分野，第 10 节会再回到它。

## 6. 插曲：V-JEPA、LeJEPA 在这张地图上的位置

到这里，语言侧的故事告一段落：第 1–5 节从"压缩的时机"出发得到四层记忆模板，并在 LLM 的 attention 上完成了落地。后半篇要把**同一个模板迁移到视觉世界模型**。

但一跨进视觉领域，就会迎面撞上一堆名字——V-JEPA、LeJEPA、Dreamer/RSSM——它们常和 Transformer/RNN/SSM 被摆在一起比较，仿佛是同一类东西的竞品。其实不是：它们根本不在同一个分类轴上。在动手拼装架构（第 13 节）之前，必须先把这些范式各自钉到正确的坐标上，否则后面一定会概念打架。

这一节先处理最容易被误解的 JEPA 系。V-JEPA 和 LeJEPA 首先是**表征学习 / latent prediction 范式**，回答的是"latent 空间如何学习"，而不是"历史如何随时间压缩成状态"——因此它们与 Transformer/RNN/SSM 不在同一个分类轴上。

### 6.1 V-JEPA：计算上像 Transformer，目标上像状态压缩

V-JEPA 的基本形式是用可见的视频区域预测被遮挡区域的 latent 表示：

$$\hat{z}_{\text{target}} = P(z_{\text{context}},\, m_{\text{target}})$$

从**计算结构**看，它接近 Transformer 一侧：使用时空 token、保留分布式上下文、通过 attention 建立关系、不要求单一递归 hidden state。

但从**学习目标**看，它又有状态压缩的倾向：预测目标是 latent 而非像素，encoder 因此被鼓励保留对预测有用的信息、丢弃难以预测或不重要的像素细节——这是一种由目标函数驱动的隐式压缩。

所以 V-JEPA 更准确的定位是：

$$\boxed{\text{基于 token 的分布式预测状态}}$$

——既不是经典单状态 RNN，也不是无压缩的原始表示。

### 6.2 LeJEPA：管表示空间，不管时间

LeJEPA 关注的是另一组问题：latent 表示如何避免坍塌、latent 空间应有怎样的统计与几何结构、如何构造稳定可扩展的 joint-embedding 目标。它**不规定**时间动力学由 Transformer、RNN 还是 SSM 实现。

因此正确的分工是：

$$\boxed{\text{LeJEPA 决定表示空间如何学习}}$$

$$\boxed{\text{Transformer/RNN/SSM 决定表示如何随时间传播}}$$

两个轴正交——这意味着它们可以自由组合，这正是后文架构方案的基础。

## 7. Observation ≠ State：世界模型必须拆分感知与信念

### 7.1 单帧表示缺了什么

进入视觉世界模型。单帧 encoder 的输出：

$$z_t = E(o_t)$$

只是**当前观测的表示**（observation representation），而不是完整的动力学状态。一帧图像原则上无法包含：

- **速度**（需要至少两帧才能定义）；
- **被遮挡的物体**（当前看不见，但确实存在）；
- **接触力**（视觉上不可见的物理量）；
- **历史动作**（agent 刚刚做过什么）；
- **当前任务阶段**（进行到哪一步了）；
- **其他未观测的隐变量**。

把 $z_t$ 直接当作状态去预测未来，等于假设世界是马尔可夫且完全可观测的——对真实视觉世界这两条都不成立。

### 7.2 Belief state 与两个必须区分的过程

因此需要在观测表示之上维护一个**信念状态**（belief state），递归地融合观测流与动作流：

$$s_t = F(s_{t-1},\, z_t,\, a_{t-1})$$

基于它预测未来——可以预测下一个状态，也可以预测下一个观测的 latent：

$$\hat{s}_{t+1} = T(s_t,\, a_t) \qquad \text{或} \qquad \hat{z}_{t+1} = P(s_t,\, a_t)$$

这里隐含着两个在概念上必须区分的过程：

1. **State estimation / filtering**：新观测到来时，用它**纠正**内部状态——"我以为杯子在桌上，现在看到它掉地上了，更新信念"；
2. **State transition / imagination**：没有未来观测时，纯靠内部动力学**向前推演**——"如果我推它一下，它会滑到哪里"。

前者有真值可对齐，后者没有；前者是感知，后者是想象。二者在概念上必须分开，但在工程上可以共享网络和参数——这正是下一节 RSSM 的做法。

## 8. Dreamer / RSSM：这个拆分的经典实现

Dreamer 系列的 RSSM（Recurrent State-Space Model）正是第 7 节拆分的标准答案，值得单独展开。

### 8.1 结构

RSSM 维护两个耦合的状态：

$$h_t = f(h_{t-1},\, z_{t-1},\, a_{t-1})$$

$$z_t \sim q(z_t \mid h_t,\, o_t)$$

其中 $h_t$ 是**确定性**递归历史状态（承载长程记忆），$z_t$ 是**随机** latent state（承载当前时刻的不确定性），$q$ 是看到真实观测 $o_t$ 之后的 posterior。

### 8.2 Prior 与 posterior：想象与感知的形式化

关键设计在于：imagination 阶段没有未来观测可用，此时模型改用不依赖观测的 prior 采样：

$$z_t \sim p(z_t \mid h_t)$$

于是两个过程获得了精确的概率对应：

$$\boxed{q(z_t \mid h_t, o_t) = \text{posterior state estimation（感知：观测修正信念）}}$$

$$\boxed{p(z_t \mid h_t) = \text{prior prediction / imagination（想象：无观测推演）}}$$

Dreamer 训练的关键就是**让 prior 与 posterior 对齐**（KL 项）：迫使模型仅凭内部动力学（prior）就能预测出接近"看了观测之后"（posterior）的状态。对齐做好了，模型就能从真实观测驱动的状态无缝切换到无观测的 latent rollout——这正是"在想象中规划"的能力基础。

### 8.3 JEPA 与 Dreamer 的互补组合

回到第 6 节的正交性，可以拼出一个清晰的分工：

- **JEPA / LeJEPA**：负责学习可预测、语义化、抗坍塌的 observation latent——决定 $z$ 空间的质量；
- **RSSM**：负责 belief state、action-conditioned dynamics 和 prior/posterior imagination——决定状态如何随时间演化；
- **actor-critic**：在想象出的 latent 轨迹上学习策略。

三者各管一层，互不冲突。这是后文完整架构的骨架。

## 9. Object-Centric World Model 的四层混合记忆

### 9.1 同一个困境,换了模态

现在到达正题。为 object-centric world model 设计记忆时，会遇到与第 4 节完全同构的困境：

- 保存**所有历史 object tokens**？内存与计算随时间无界增长，等价于全量 KV 的代价；
- 把所有对象历史**压成一个 global state**？不同对象的信息在固定容量中互相干扰，等价于单一递归 summary 的失败模式。

答案同样是分层——但对象世界给了我们比文本更强的结构先验（对象持续存在、事件离散发生、背景相对静态），因此可以分得更有物理意义：

$$\boxed{\text{短期对象轨迹} + \text{长期对象状态} + \text{稀疏事件记忆} + \text{全局场景记忆}}$$

完整状态：

$$S_t = \left(M_t^{\text{local}},\; \{h_t^i\}_{i=1}^{K},\; M_t^{\text{event}},\; g_t\right)$$

下面逐层展开。

### 9.2 第一层：短期精细 Object Memory

保留最近 $W$ 帧的完整 object tokens：

$$M_t^{\text{local}} = \{O_{t-W+1},\, \ldots,\, O_t\}, \qquad O_t = \{o_t^1, \ldots, o_t^K\}$$

每个 object token 携带结构化的属性组：

$$o_t^i = \left[\, z_t^i,\; p_t^i,\; v_t^i,\; r_t^i,\; c_t^i,\; u_t^i \,\right]$$

分别是：视觉/语义 latent、位置、速度、姿态、置信度、不确定性。

这一层服务于所有"还没到可以安全压缩的时候"的信息：

- 短期动力学（精确的运动微分）；
- 精确轨迹；
- 接触瞬间（碰撞前后几帧的细节决定物理推断的成败）；
- 跨帧的 object correspondence；
- 近期发生、意义尚不明朗的变化。

它对应语言模型中的 local full-resolution window（第 5.1 节）。

### 9.3 第二层：长期 Object State

为每个持续存在的对象维护一个递归 belief state：

$$h_t^i = F_{\text{obj}}\!\left(h_{t-1}^i,\; o_t^i,\; m_t^{i,\text{interaction}}\right)$$

（其中 $m_t^{i,\text{interaction}}$ 是来自其他对象的交互消息，见第 11 节。）它保存那些跨越短窗口、必须持续追踪的内容：

- object identity（这是同一个杯子）；
- 长期运动趋势；
- **遮挡期间的状态**——对象暂时看不见时，belief state 是它存在的唯一凭据；
- 稳定物理属性（质量感、材质、刚度）；
- 历史交互留下的持续变化（杯子已经空了、门已经开了）。

一个有用的细化是按时间尺度拆分：

$$h_t^i = \left(h_t^{i,\text{fast}},\; h_t^{i,\text{slow}}\right)$$

- **fast state**：位置、速度、接触、短期动态——每帧都变；
- **slow state**：身份、材质、质量、刚度、任务角色——几乎不变，一旦确立就该被保护起来，避免被高频更新冲刷。

这一层就是 object 级的 RNN/SSM/RSSM——第 1 节中"把历史变成状态"的能力，在对象粒度上恰得其所：单个对象的状态维度需求是有限的，递归压缩在这里不再盲目。

### 9.4 第三层：稀疏 Event Memory

一个关键观察：**长期来看真正重要的历史往往不是某个静态状态，而是离散的事件**。"杯子在 3 分钟前被从桌上拿到了柜子里"这条信息，用逐帧轨迹表达极其冗余，用一条事件记录表达恰到好处。

事件的结构化表示：

$$e_j = \left(\text{participants},\; \text{event type},\; \text{time},\; \text{pre-state},\; \text{post-state}\right)$$

典型事件类型：抓取、释放、碰撞、支撑关系改变、遮挡开始/结束、柜门打开、物体被移入新容器。

**写入判据**是这一层的设计核心——不能什么都写（退化为逐帧存储），也不能漏掉关键转折。写入分数综合四个信号：

$$g_t = \alpha\,\text{prediction surprise} + \beta\,\text{state change} + \gamma\,\text{interaction strength} + \delta\,\text{task relevance}$$

直觉是：**预测误差高**（世界出乎意料）或**关系发生突变**（拓扑改变）的时刻，正是值得记住的时刻。只有分数超过阈值才写入：

$$M_t^{\text{event}} = M_{t-1}^{\text{event}} \cup \{e_t\}$$

这一层对应语言模型的稀疏检索层（第 5.2 节）：稀疏、按需、保留因果关键点。

### 9.5 第四层：全局 Scene Memory

对象并不能解释整个视觉世界——背景、几何、相机自身的运动都在对象之外。因此需要全局场景状态：

$$g_t = F_{\text{scene}}\!\left(g_{t-1},\; O_t,\; \text{camera}_t\right)$$

它保存：

- 静态背景；
- 地图与场景几何；
- 相机 / agent 位姿；
- 全局空间拓扑（哪个房间连着哪个房间）；
- 当前任务阶段；
- 以及一个诚实的兜底项：**无法稳定 object 化的残余信息**——对象分解永远不完美，残余需要有处安放，否则会污染对象表示。

实现形式可以多样：3D feature map、BEV memory、tri-plane、sparse voxel map，或最简单的一组 global scene tokens。

它对应语言模型的全局低分辨率、始终可见的摘要层（第 5.3 节）。

## 10. 读取：四层记忆如何协同服务一次预测

分层存储只有配上分层读取才完整。预测对象 $i$ 的未来时，用它的 query $q_t^i$ 分别读取四层：

$$c_t^{i,\text{local}} = \operatorname{Attn}\!\left(q_t^i,\; M_t^{\text{local}}\right) \qquad \text{（近期精确变化）}$$

$$c_t^{i,\text{object}} = h_t^i \qquad \text{（自身长期状态，直接可用无需检索）}$$

$$c_t^{i,\text{event}} = \operatorname{TopKRetrieve}\!\left(q_t^i,\; M_t^{\text{event}}\right) \qquad \text{（相关历史事件，稀疏检索）}$$

$$c_t^{i,\text{scene}} = \operatorname{Attn}\!\left(q_t^i,\; g_t\right) \qquad \text{（全局背景与空间约束）}$$

再融合为增强状态：

$$\tilde{h}_t^i = \operatorname{Fuse}\!\left(h_t^i,\; c_t^{i,\text{local}},\; c_t^{i,\text{event}},\; c_t^{i,\text{scene}}\right)$$

四层的分工一目了然：

- **local memory**：这个对象最近几帧的精确变化；
- **object state**：它是谁、长期处于什么状态；
- **event memory**：它经历过哪些因果关键点；
- **scene memory**：它身处怎样的全局环境与几何约束。

这与第 5 节 LLM 三层结构的对应关系是精确的：local / event / scene 三层正是那套"读取侧"分层（recent / 稀疏检索 / 全局摘要）在对象模态下的实例，而 **object state 是新增的一层**。回到 5.5 的区分——DeepSeek 式 sparse attention 只做到了访问方式的分层，节点仍是原始 token；这里的 $h_t^i$ 则是显式造出的**新节点、新状态**，即那个一直缺席的"表示侧"。对象世界之所以能补上这一层，是因为它比文本多了一个"持续存在的实体"这一结构先验：单个实体的状态维度需求有限，递归压缩在此不再盲目（第 9.3 节）。

## 11. Object Interaction：状态演化不能各自为政

### 11.1 为什么必须显式建模交互

对象的状态变化经常**来自交互**而非自身惯性：被推动、被支撑、被碰撞、被抓起。如果每个 object state 独立更新，模型将无法解释"A 撞了 B，所以 B 动了"——这恰恰是物理理解的核心。

### 11.2 Graph attention 消息传递

用图注意力在对象之间传递交互消息。对象 $i$ 收到的消息：

$$m_t^i = \sum_{j \ne i} \alpha_{ij}\, \phi\!\left(h_t^i,\; h_t^j,\; r_{ij}\right)$$

注意力权重由状态相容性和**关系特征**共同决定：

$$\alpha_{ij} = \operatorname{softmax}_j\!\left(q_i^\top k_j + b(r_{ij})\right)$$

关系特征 $r_{ij}$ 编码了物理交互的先验，包括：

- 相对位置；
- 相对速度；
- 接触状态；
- 支撑关系（谁托着谁）；
- 遮挡关系；
- 是否正被同一个 agent 操作。

这些特征作为 attention 的 bias 项 $b(r_{ij})$，让"接触中的对象对"天然获得更强的信息通道。

### 11.3 融入状态更新

交互消息进入下一步的状态更新：

$$h_{t+1}^i = F\!\left(h_t^i,\; o_{t+1}^i,\; m_t^i,\; a_t\right)$$

于是整个动力学模型的本质结构是：

$$\boxed{\text{per-object state evolution} + \text{interaction graph}}$$

——每个对象有自己的演化方程，方程之间通过一张动态的交互图耦合。

## 12. Object Identity：整个系统的前提

### 12.1 绑定问题

有一个前提条件贯穿以上所有设计：当前观测中检测到的 object 必须与历史 object state **正确配对**：

$$o_t^i \;\leftrightarrow\; h_{t-1}^j$$

配错一次，此后所有的状态更新都在污染错误的实体——"杯子的历史"被写进了"碗的状态"。这个绑定问题必须显式解决，**不能依赖固定的 slot index**（第 3 个检测框 ≠ 第 3 个对象：检测顺序会变、对象会进出视野）。

### 12.2 匹配机制

构造外观与几何联合的匹配分数：

$$A_{ij} = \operatorname{sim}\!\left(o_t^i,\; \hat{h}_t^j\right) + \lambda_p\, \operatorname{geom}\!\left(p_t^i,\; \hat{p}_t^j\right)$$

第一项衡量外观/语义相似度，第二项衡量位置一致性（对象不会瞬移）。然后用标准分配算法完成关联：

- soft assignment（可微，适合端到端训练）；
- Sinkhorn（可微的近似双随机匹配）；
- Hungarian matching（离散最优分配）；
- 或 attention-based tracking。

### 12.3 更自然的流程：predict-then-match

注意上式中的 $\hat{h}_t^j$ 带帽子——这暗示了正确的时序。比起拿当前观测与**上一帧**状态硬匹配，更自然的流程是先让 dynamics 把所有对象状态**预测到当前时刻**：

$$\hat{h}_t^j = T\!\left(h_{t-1}^j,\; a_{t-1}\right)$$

再拿当前观测与这些 prior 预测匹配、并做修正：

$$h_t^j = U\!\left(\hat{h}_t^j,\; o_t\right)$$

这个 predict → match → correct 的循环，正是经典目标跟踪中卡尔曼滤波"预测-关联-更新"流程的神经版本；用第 8 节的语言说，它就是 **object-level 的 Dreamer/RSSM**：prior 预测（想象对象现在应该在哪）+ posterior 修正（观测告诉我它实际在哪）。快速运动、短暂遮挡下的匹配鲁棒性由此而来——匹配的对象是"预测位置"而不是"旧位置"。

## 13. 完整架构：把所有组件串成一个循环

现在把第 6–12 节的组件组装成完整的每步计算流程。

**① Observation Encoder** —— 感知前端：

$$X_t = E(o_t)$$

$X_t$ 是视觉 patch tokens（encoder 可用 JEPA 式预训练目标学习，见第 14 节）。

**② Object Binding** —— 从 patch 到对象：

$$O_t = B\!\left(X_t,\; \hat{H}_t\right)$$

关键细节：binding 以 prior object states $\hat{H}_t$ 为条件——模型带着"我预期这些对象在这些位置"的先验去解析画面，而不是每帧从零分割。这同时完成了检测与关联的耦合。

**③ Prior Dynamics** —— 想象一步：

$$\hat{h}_t^i = T_{\text{prior}}\!\left(h_{t-1}^i,\; a_{t-1},\; m_{t-1}^i\right)$$

注意交互消息 $m_{t-1}^i$ 在其中——想象也要考虑对象间的相互作用。

**④ Posterior Correction** —— 观测修正：

$$h_t^i = T_{\text{post}}\!\left(\hat{h}_t^i,\; o_t^i\right)$$

**⑤ Event Write** —— 稀疏事件判定：

基于三元组 $\left(h_{t-1}^i,\; \hat{h}_t^i,\; h_t^i\right)$ 计算 prediction residual（prior 与 posterior 的差距，即"惊讶度"）、state change 和 interaction change，按第 9.4 节的判据决定是否写入 event memory。注意这个设计的巧妙之处：**惊讶度信号是 prior/posterior 结构免费送的副产品**。

**⑥ Scene Update** —— 全局状态推进：

$$g_t = F_{\text{scene}}\!\left(g_{t-1},\; O_t\right)$$

**⑦ Future Rollout** —— 纯想象模式：

规划或预测未来时没有观测，跳过 ②④⑤，只用 prior dynamics 迭代推进：

$$\hat{h}_{t+1}^i = T_{\text{prior}}\!\left(h_t^i,\; a_t,\; m_t^i\right)$$

并在需要时读取 event memory 与 scene memory（第 10 节）。感知模式（①–⑥ 全跑）与想象模式（只跑 ③+⑦）共享全部参数——这正是 prior/posterior 对齐训练要保证的能力。

## 14. 训练目标

六个损失项，各自锚定架构的一个组件：

**① Object Latent Prediction** —— 主预测损失：

$$\mathcal{L}_{\text{pred}} = \sum_i d\!\left(\hat{o}_{t+1}^i,\; o_{t+1}^i\right)$$

目标定义在 latent 空间而非像素空间（JEPA/LeJEPA 风格）——模型不必预测每根草的摆动，只需预测对象的语义状态。

**② Prior–Posterior Alignment** —— 想象能力的来源：

$$\mathcal{L}_{\text{KL}} = D_{\mathrm{KL}}\!\left[\, q(h_t^i \mid o_t)\; \|\; p(h_t^i \mid h_{t-1}, a_{t-1}) \,\right]$$

迫使"不看观测的预测"逼近"看了观测的估计"，使 rollout 可信（第 8.2 节）。

**③ Identity / Association Consistency** —— 约束同一对象跨时间保持稳定 identity，防止绑定漂移（第 12 节的训练侧保障）。

**④ Relation Prediction** —— 辅助任务，显式预测 contact、collision、support、occlusion、relative motion，迫使交互图（第 11 节）学到真实的物理关系而非退化为均匀注意力。

**⑤ Multi-Step Rollout** —— 长程稳定性：

$$\mathcal{L}_{\text{rollout}} = \sum_{\tau=1}^{H} \lambda_\tau\, d\!\left(\hat{O}_{t+\tau},\; O_{t+\tau}\right)$$

单步预测好不代表多步不发散；多步损失直接惩罚误差累积。

**⑥ Event Sparsity** —— 写入门控的正则：

$$\mathcal{L}_{\text{sparse}} = \sum_t \left\|g_t^{\text{write}}\right\|_1$$

没有它，写入门会学会"全都写"（写入永远不亏预测损失），event memory 退化为逐帧存储。

## 15. 最小可行版本（MVP）：从哪里开始

第 13–14 节的完整系统组件很多，不建议一次实现。以下配置保留了每一层的最小形态，可以先跑通再逐项升级：

1. 每帧提取 $K$ 个 object tokens（$K$ 取场景典型对象数）；
2. 短期记忆只保留最近 **8–16 帧**的完整 object tokens；
3. 每个 object 用一个 **GRU 或简单 SSM** 作为 hidden state（先不拆 fast/slow）；
4. 全局场景用 **1–4 个 global scene tokens**（先不上 BEV/体素）；
5. 交互用 **2–4 层 graph attention**；
6. 事件写入判据先只用一条：**latent prediction residual 超过阈值**（先不学四项加权门控）；
7. rollout loss 用 **4–8 步**；
8. 加上 object association consistency 约束；
9. **先做 deterministic dynamics，跑通后再引入 stochastic latent**（完整的 prior/posterior 分布是第二阶段的事）。

最小结构可以概括为：

$$\boxed{\text{local object window} + \text{per-object recurrent state} + \text{global scene token} + \text{sparse surprise events}}$$

——四层记忆一个不缺，但每层都取最简实现。

## 16. 结论：一个统一原则

整个讨论收敛到一条原则：

> **世界模型不应在"保留全部历史"和"压成单一状态"之间二选一，而应根据不同信息的时间尺度、可压缩性、精确性要求和未来可查询性，采用多层记忆。**

同一个模板在两个领域的实例化：

**语言模型：**

$$\boxed{\text{近期完整 KV} + \text{中尺度稀疏检索} + \text{全局粗粒度摘要}}$$

**Object-centric world model：**

$$\boxed{\text{近期对象轨迹} + \text{长期对象 belief state} + \text{稀疏因果事件} + \text{全局场景记忆}}$$

四层对象记忆各自的存在理由，一句话概括：

- **Object state** 保存持续存在的实体——世界的"名词"；
- **Event memory** 保存稀疏但因果重要的变化——世界的"动词"；
- **Scene memory** 保存全局结构和无法 object 化的背景——世界的"舞台"；
- **Local memory** 保存尚未安全压缩的精细信息——判断的"缓冲期"。

最终最值得探索的架构，是三条线索的合流：

$$\boxed{\text{Object-Centric RSSM/SSM} + \text{Hierarchical Attention Memory} + \text{JEPA-Style Latent Prediction}}$$

它同时继承了五个范式各自最强的部分：

- **Transformer** 的按需检索能力（第 1、5、10 节）；
- **RNN/SSM** 的持续状态建模能力——用在对象粒度上恰得其所（第 9.3 节）；
- **Dreamer** 的 prior/posterior imagination 机制（第 8、12、13 节）；
- **JEPA/LeJEPA** 的预测性表征学习（第 6、14 节）；
- **object-centric model** 的实体化、结构化世界表示（第 9–12 节）。

没有任何单一范式能同时提供这五种能力；而一旦承认"不同信息需要不同的记忆策略"，它们的组合方式几乎是被这条原则唯一确定的。
