### 任务总述

我现在在构建一个算法创新类型科研的辅助实验代码，基于 WAM 中 Flow Matching 的推理步骤进行实时反馈的优化，详细算法原理见后文。辅助实验最终目的是为了验证我们的 FBFM算法在模型推理预测的 state 准确性上相较于传统的 RTC 和 FM 是有提升的。核心项目流程如下，请为我逐项完成：

1. 第一步：构建一个轻量型的动力学仿真环境，可以考虑一维的“质量-弹簧-阻尼器 MSD”提供一组比较适中的参数 $(m,c,k,dt)$，状态量考虑二维的 $s=[x,\dot{x}]$ 表示位置和速度，控制量考虑一维的 $a=[u]$ 表示施加的力。同时需要注意，我要在数据集中人为添加相关的扰动噪声，【添加噪声的位置我没想好】
2. 第二步，使用 PID 控制器自己构建专家数据，并使用流匹配 DiT 进行模型训练。
3. 第三步，因为我的算法的优势【可以结合下文的算法为我补充一下这点】，我希望针对这种优势及其应用场景，设计相应的扰动实验，可以包括：在动作执行的一个 chunk 内突然给予扰动、或是直接在测试时使用与仿真环境不同的参数等等方式【请结合算法的优势为我设计一下，将详细的设计细节列在这里】
4. 第四步，按部就班，在确定好实验设计的原理之后，我希望你能在一个 python 文件中分别准确地实现下文中谈到的三个算法，并在合适的实验环境中进行运行，并记录输出数据。
5. 第五步，需要注意对输出数据的利用方式，我需要做的可视化能够凸显出我的算法的优势，所以请生成一份代码文件，使用科研论文级别的好看的高信息量、高饱和度配色的方式生成相应的可视化图片。

### 详细算法实现

#### 对照一：传统 Flow Matching
**首先，明确一下Flow Matching的算法概述**，所谓流匹配，简明扼要地讲，就是学习一个固定步骤的去噪过程，我的模型通过神经网络维护一个向量场，在VA中的输入是每一个时刻读取到的当前状态，输出是后续的另一状态，经过VAE的Decoder将其应射成为一个长度的 $H$ 的动作序列。在 $a_{-1}$ 执行过之后读取状态 $o_t$ 作为输入，输出动作序列 $\{a_1 \cdots a_{16}\}$.

#### 对照二：Real-Time Chunking, RTC 算法实现细节
**其次，明确一下RTC在流匹配输出时候的作用和流程**。简明扼要地讲，RTC，Real-Time Control起到了对于流匹配向量场的一个实时的补充和维护的作用。(详细算法原理与解释见同路径下的“RTC Paper.pdf”文件)

- 首先，在$a_{-1}$执行过之后读取状态$o_t$作为输入，随后我的Flow Matching经过四个时间步的推理，输出一个长度为$H$的动作序列$\{A_t\}_{1\cdots16}$，需要注意的是，推理过程会依赖前一个动作序列$\{A_{t-1}\}_{1\cdots16}$中所有未被执行的量，即$\{a_0,\cdots,a_{10}\}$，不放设其为$A_{prev}$随后，我们会利用其与模型输出的$H$维的张量进行梯度下降维护流场，也就是在每一步去噪的过程中进行维护。需要注意的是，因为$A_{prev}$和预测输出$Y$维度不同，我们会认为用0进行维度补齐，同时，我们会通过掩码机制，对于后补的几项不进行梯度更新，以避免对其误导。

- 其次对于输出的序列，不妨设有16位，其中前四位$\{a_0,a_1,a_2,a_3\}$为推理时刻同时异步产生的，也就是上一个序列$\{A_{t-1}\}_{1\cdots16}$中的第六到九项，此时可以将其理解为一个盲目执行的Action Chunk。随后五位是一定会被执行器闭着眼睛执行的，在$a_4$执行结束之后，系统会读取当前的状态$o_{t+1}$，作为下一刻流匹配的输入，同时读取$\{A_t\}_{1\cdots16}$中未被执行的量，也即$\{a_5,\cdots,a_{15}\}$，补全后作为下一次梯度更新的目标。

- 所以总体上来看，每一次推理，是可以通过实时地更新流场来将输出的动作与先前的动作更加流畅地衔接起来，但是在推理的同时，执行器却会异步地、没有任何感知的执行写死的程序，一旦在这个过程中环境突然发生突变，执行器没有进行实时调整的可能。所以如果预测不准确，还是有可能会出现轻微的锯齿，但是整体上看是比较流畅连续的。

#### 实验组：Feed-Back Flow Matching, FBFM 算法实现
**最后，关于我们改进的算法FBFM，Feed Back FLow Matching**，它主要实现了在推理和执行的异步过程中，能够实时地处理读取到的数据，并避免盲目地执行丁巳的程序这一问题。模型整体的框架思路和RTC是类似的，只不过我们做了如下几点改动：

- 首先，我们将状态变量$X^{\tau}_T$更新成了$X^\tau_t=[Z^\tau_t,A^\tau_t ]$，在每一次记录模型输入的时候，不光读取当前的state，同时还会读取当前的action，同时作为模型的输入。

- 其次，更重要的时，我们对向量场的梯度更新方式有改变，我们不再用类似RTC中的现预测序列与前置预测序列进行趋近的方法，而是当我的执行器每执行一次，我就会直接读取当前的 $X^\tau_T$ 作为输入，经过一步流匹配的计算得到一个，基于当前模型预测出来的动作序列。$$\hat{X^1_t}=X^\tau_t+(1-\tau)v(X^\tau_t,o_t, \tau)$$ 同时，仿照RTC中对于目标值的处理，我们从上一个动作序列中最终获取的动作和状态，同样补全后构成 $Y$，同样利用梯度反馈机制进行更新：$$v_{\Pi GDM}(X^\tau_t,o_t, \tau)=v(X^\tau_t,o_t, \tau)+k_p(Y-\hat{X^1_\tau})^Tdiag(W)\frac{\partial\hat{X^1_t}}{\partial{X^\tau_t}}$$ 其中：
	- $\hat{X^1_t}=X^\tau_t+(1-\tau)v(X^\tau_t,o_t, \tau)$
	- $k_p=min(\beta,(\frac{1-\tau}{\tau})(\frac{1}{r_\tau^2}))$
	- $r_\tau^2=\frac{(1-\tau)^2}{\tau^2+(1-\tau)^2}$

- 所以总体上来看，我们的推理链条和执行链条时可以产生实时地交互的，也就是每一次我的控制器执行完一次动作，都会直接跟已经去噪了若干步的Flow Matching模型进行同步与修正，也就是会起到一个实时纠偏的过程，避免了chunk之间的锯齿问题。
---
Algorithm: FBFM

**Require:**

- $\pi$: Flow policy
- $H$: Prediction Horizon
- $s_{chunk}$: Minimum Chunk Execution Horizon
- $s_{step}$: Minimum Step Execution Horizon
- $\mathcal{M}$: Mutex (用于同步 `GETACTION` 和 `INFERENCELOOP`)
- $\mathcal{C}$: Condition Variable
- $A_{init}$: Initial Action Chunk
- $d_{init}$: Initial Delay Estimate
- $b$: Delay Buffer Size
- $n$: Number of Denoising Steps
- $\beta$: Maximum Guidance Weight
- $\mathcal{PC}$: RTC Previous Chunk (用于上下文一致性)

**Procedure 1: `INITIALIZESHAREDSTATE`**

1. $t = 0; A_{cur} = A_{init}; o_{cur} = null$
2. Initialize $\mathcal{PC}$ from $A_{cur}[s, s+1, \dots, H-1]$

**Function: `GETACTION`($o_{next}$)**

3. **with** $\mathcal{M}$ acquired **do**
4. $\quad t = t + 1$
5. $\quad o_{cur} = o_{next}$
6. $\quad$ notify $\mathcal{C}$
7. **with** $\mathcal{M}$ released **do**
8. $\quad$ encode $o_{cur}$ as $z_{cur}$
9. $\quad \mathcal{PC}.append(\text{state latent}(z_{cur}))$
10. **return** $A_{cur}[t-1]$

**Procedure 2: `INFERENCELOOP`**

11. acquire $\mathcal{M}$
12. $Q = \text{new Queue}([d_{init}], \text{maxlen} = b)$
13. **loop**
14. $\quad$ wait on $\mathcal{C}$ **until** $t \ge s_{chunk}$
15. $\quad s = t$
16. $\quad$ new $\mathcal{PC}$ from $A_{cur}[s, s+1, \dots, H-1]$
17. $\quad o = o_{cur}$
18. $\quad d = \max(Q)$
19. $\quad$ **with** $\mathcal{M}$ released **do**
20. $\quad \quad A_{new} = \text{GUIDEDINFERENCE}(\pi, o, A_{prev}, d, s)$
21. $\quad A_{cur} = A_{new}$
22. $\quad t = t - s$
23. $\quad$ enqueue $t$ onto $Q$

**Function: `GUIDEDINFERENCE`($\pi, o, A_{prev}, d, s$)**

24. get $\mathbf{W}$ from $\mathcal{PC}$
25. get $X_{prev}$ from $\mathcal{PC}$ and right-pad $X_{prev}$ to length $H$
26. initialize $X^0 \sim \mathcal{N}(0, \mathbf{I})$
27. **for** $\tau = 0$ **to** $1$ **with step size** $1/n$ **do**
28. $\quad f_{\hat{X}^1} = \mathbf{X}' \mapsto \mathbf{X}' + (1-\tau)\mathbf{v}_\pi(\mathbf{X}', \mathbf{o}, \tau)$
29. $\quad e = (X_{constrain} - f_{\hat{X}^1}(X^\tau))^\top \text{diag}(\mathbf{W})$
30. $\quad \mathbf{g} = \mathbf{e} \cdot \frac{\partial f_{\hat{X}^1}}{\partial \mathbf{X}'} \Big|_{\mathbf{X}' = X^\tau}$
31. $\quad X^{\tau + \frac{1}{n}} = X^\tau + \frac{1}{n} \left( \mathbf{v}_\pi(X^\tau, \mathbf{o}, \tau) + \min(\beta, \frac{1-\tau}{\tau \cdot \mathbf{r}_\tau^2})\mathbf{g} \right)$
32. **return** $A^1$
---
### 详细项目要求

1. 我希望代码是读者友好型的方式编写，注释丰富，调用结构清晰，变量命名准确，每一个重要张量都标注其 shapes。
2. 我希望你可以为我构建算法外围的所有程序代码，但是 FBFM 与 RTC 最核心的实现函数实现，是由我自己编写，我希望你在相应的函数中使用注释告诉我需要用到的相关接口并提示我应该如何写、告诉我相关联的代码位置。但是最终的视线会由我来进行。
3. 需要注意的一点是，我在此会采用单线程来模拟推理链与执行链的异步双线程实现，结合上述的算法实现内容可知：
	- 在 RTC 中我可以进行完五步 (第一步后读取的 state 数据作为 flow matching 的输入、后面四步正常执行) action 的执行【执行链】 $\rightarrow$ 执行完整的带反馈的 Flow Matching【推理链】 $\rightarrow$ 下一周期的执行。
	- 在 FBFM 中我则是使用了实时的状态反馈：
		- 推理链先执行若干步骤的去噪，假设是四步
		- $\rightarrow$ 切换到执行链完成一个动作，把当前的 state 和 action 提取传回去作为 Flow Matching 实时梯度反馈的数据
		- $\rightarrow$ 切换到推理链进行更新流场，并完成下一个小周期的去噪，假设是四步
		- $\rightarrow$ 切换到执行链下一个动作，如此循环直到推理的去噪过程完整结束
4. 我希望你针对上面的文本，为我生成一份由你实时更新完善的 `CLAUDE.md` 文档，记录本项目的所有注意事项、要求规范、模块化完成进度、实验设计细节、优化评估方向和对话框中最核心最重要的结论。每一次上下文更新都会读取这个文件，从这一个文件中就能了解项目的全部进度和细节。同时我想强调，我不希望你一下就大而全地生成所有的代码，我希望你结合前文中的思路，将整体的工程拆分成若干个可以验证正确性的模块任务，每一次只执行一个并确保其清晰和准确。