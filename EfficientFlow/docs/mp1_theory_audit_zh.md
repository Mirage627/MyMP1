# MP1 训练统计量方案：桌面 HTML / PDF 审核

审核日期：2026-10-30。本文只记录算法解释与源文件纠错，不修改源 HTML/PDF，不把理论可计算性当作已获得机器人收益。

## 1. 核对来源与实施范围

- 桌面 HTML：`C:/Users/67223/Desktop/MeanFlow_代码前理论规范_v2 (1).html`，SHA256 `E80577EB812C2994CCA0071AB1215B2352C70830D0869C4716FE6BF9C1DF4E5B`。
- 桌面一页 PDF：`C:/Users/67223/Desktop/理论 (1).pdf`，SHA256 `42438E0297BE065E7F37F2DB96E447BB38E714CC9879249E34A2E2AD6CD7783C`。已提取全文并渲染整页核对公式。
- 原始 v2 Markdown：`C:/Users/67223/.codex/visualizations/2026/10/28/01a180ba-9f2b-7320-b872-a4f303aea6d4/MeanFlow_代码前理论规范_v2.md`，SHA256 `CC1212BF4F88CBAF8D0C51C911E9B4563F734E37049952889B81FF4491B63A6E`。
- 已读取关联 chat `01a180ba-9f2b-7320-b872-a4f303aea6d4` 最新三个回合。最终讨论选择 H5a 完整矩阵作为研究主候选，H5b 对角作为必要对照；保留区间统计量、有界度量及分离更新，尚未宣称实际收益。

HTML 保留了原公式与其下的 H/B 修订。执行时以“对应修改 H”作为 H 算法规范，不能把保留供对照的原双 Jacobian 估计器又叠入 H。PDF 是 H 的简化摘要，存在下述省略及错误，不能逐字作为无歧义实现契约。

本次用户已授权实现。旧文档“先低维实验再进入机器人”的研究顺序建议及“未证明普遍提升”的说明，应保留为结果解释和验收边界；它们不是停止已授权编码或重复请求许可的理由。实现与有意义的数值/梯度/优化器验证可以继续。缺少训练实验时，仍不得报告性能已提升。

## 2. PDF 逐公式纠错

### (1) Label：形式正确，补齐冻结和时间约定

令 `z=(x_t,t,r,o)`、`h=t-r`、`d=epsilon-a`，数据端为 0，噪声端为 1。标签应为

\[
y_{\bar\theta}=d-h\operatorname{JVP}_{(x,t,r)}\bar u[(d,1,0)],\qquad
e_\theta=u_\theta(z)-\operatorname{sg}(y_{\bar\theta}).
\]

`partial_t` 固定 `x,r,o`；不能在求时间导数时令 `r=tau` 跟随 `t=tau`，否则多出 `partial_r`。teacher 参数、buffer 及条件随机性冻结，但仍允许对输入 x、t 求导；不能用禁用全部输入导数的上下文替代参数冻结。构建完成的 y 停止梯度，student 输出保留参数梯度。该式在 w=1 时对应严格 H 标签。

### (2) Variance：主要实质错误，混淆瞬时目标与区间标签

PDF 将 `S=Cov(y)` 直接等同于 `(I-t J_x v*)/(1-t)`，遗漏区间传播矩阵。应分别定义

\[
C=\operatorname{Cov}(d\mid x_t,t,o),\qquad
A=I-hJ_x\bar u,\qquad
S=\operatorname{Cov}(y_{\bar\theta}\mid z)=ACA^\top.
\]

在线性独立高斯桥及所需正则条件下，内点的背景恒等式是

\[
C=\frac{I-tJ_xv^*}{1-t},\qquad
v^*=E[d\mid x_t,t,o].
\]

即便一维，区间标签方差仍为 `S=A^2 C`，不能删除 A。PDF 的 `Cov(y)` 也应补成条件协方差 `Cov(y|z)`。

将 `J_x v*` 近似成任意区间平均速度 `J_x u_bar(x,t,r,o)`，不是上述恒等式。若作为旧插件估计器，应使用单独定义的瞬时近似 `v_bar(x,t,o)=u_bar(x,t,t,o)` 并交代估计误差。**H 直接学习区间残差二阶矩，不执行这个插件协方差估计，也不计算两个额外完整 Jacobian。**原 MeanFlow 标签的一次 JVP 仍保留。

### (3) Prediction of head：一维写法可用，完整版本须用外积

完整矩阵头的预测对象应写成

\[
M(z)=E[e_\theta e_\theta^\top\mid z].
\]

PDF 的平方表达仅适用于一维或对角各元素。头的输入只能来自 `z` 及允许的特征，不可把当前样本隐藏动作 a、目标 d、标签 y 或残差 e 作为输入；这些量只用于监督。

### (4) M 的分解：高维外积方向需修正

采用列向量记法时，应为

\[
M=S+bb^\top,\qquad b=u_\theta-E[y_{\bar\theta}\mid z].
\]

PDF 写 `b^T b`，在高维中是标量而非 D×D 矩阵。一维时两者数值相同，所以一维简式遮蔽了此问题。头默认拟合 M，而非纯 S；只有均值偏差消失且头准确时，两者才一致。

### (5) Loss of head：补齐梯度隔离，区分参数化与损失

H 的头损失为

\[
\mathcal L_\phi^H=\frac1D E\left[
\log\det\Sigma_\phi+
\operatorname{sg}(e_\theta)^\top\Sigma_\phi^{-1}\operatorname{sg}(e_\theta)
\right].
\]

输入到统计头的共享速度/观察特征也须 detach。仅 detach 残差还不足以隔离共享编码器梯度。

\[
L_\phi=\operatorname{unpack}_\triangle(g_\phi(z)),\qquad
\Sigma_\phi=L_\phi L_\phi^\top+\delta_\Sigma I,\quad\delta_\Sigma>0.
\]

这里 `L_phi` 是因子矩阵，`mathcal L_phi` 是损失，代码中应使用不同名称。因子对角非零初始化，避免全零 L 导致 `LL^T` 的一阶导数全零。使用 Cholesky solve 与 logdet，避免显式求逆；NLL 可以为负，不应裁成零。箭头“→M”是理想总体统计最优的说明，不是每个 batch 必然恢复真实 M 的保证；SPD 下界、模型容量和有限优化均可能产生误差。

### (6) Main loss：权重必须停止梯度

\[
\mathcal L_\theta^H=\frac1D E[e_\theta^\top\operatorname{sg}(W_\phi)e_\theta].
\]

PDF 的 W 变换与 HTML H7 一致，补齐范围即可：

\[
R_\phi=(\lambda+\lambda_{\min}(\Sigma_\phi))
(\Sigma_\phi+\lambda I)^{-1},\qquad
W_\phi=I+\rho\left(R_\phi-\frac{\operatorname{tr}R_\phi}{D}I\right),
\]

其中 `lambda>0`、`0<=rho<1`。`lambda` 是度量 ridge，`delta_Sigma` 是统计头下界，不能混为一个超参。正常有限 SPD 输入下有 `tr(W)=D` 和 `(1-rho)I <= W <= (1+rho)I`，这只约束损失度量，不保证梯度方差、训练收敛或任务成功率改善。

**标量/各向同性头的特例：**若 `Sigma=qI`，则 `R=I`、`W=I`，所以严格 H 下标量组不会改变 student 的平方回归权重。它适合作为额外统计头开销的负对照；若另用标量逆方差缩放主损失，那已经是另一种权重公式，需要单独命名。

## 3. H 的执行契约

1. 完整动作 chunk 展平为 `D=horizon*action_dim`；完整头输出 `D(D+1)/2` 个下三角参数，对角头输出 D 个数，标量头输出 1 个数并扩为 qI。预测头、NLL 和 W 必须使用同一套向量化次序。
2. 复用一次 batch 中一致的观察增强与条件编码，先计算更新前的 student、teacher 标签、Sigma、W、两项损失及两组梯度，再更新互不重叠的 theta/phi 参数组。不能先用当前样本更新头，再用更新后的头重算同一样本的 W。
3. `L_theta` 不向 phi 回传；`L_phi` 不向 theta 或共享编码器回传。两组参数应显式在相应优化器中注册并检查实际更新。detach 是梯度隔离，不表示后续训练中特征数值已永久冻结。
4. `r=t` 时标签退化为 d，应精确返回对角锚定分支并跳过不必要的 JVP；严格 H 首轮令该分支 `W=I`。头可利用对角残差学习，但不把该权重加入 student 的对角锚定。
5. 端点/启用区间通过仅依赖允许输入的门函数 chi 控制，`W_used=I+chi*(W-I)`、`0<=chi<=1`。不使用 PDF 错误的 `1/(1-t)` 插件公式计算 H。保持端点基础训练，不因关闭矩阵度量而删除整个时段。
6. 记录非有限输入、矩阵分解失败及回退原因。允许在声明的策略下将权重回退 I，但不能用当前残差决定是否回退；非有限标签/NLL不能静默当作有效样本。
7. 推理仍只用 MP1 的平均速度网络；统计头不必部署，不添加 SUREFlow 的终点细化，不增加默认1NFE调用。训练增加统计头与矩阵代价，须记录训练时间、显存和分解失败率。

## 4. 与现有 MP1 的配置兼容

### H11 的损失相加与实际更新

H11 要求的是梯度归属，不要求必须调用两次 `backward()`。在统计头输入、速度损失的 W、统计损失的 e 三处停止交叉梯度后，自动微分满足

\[
\nabla_\theta(L_\theta+L_\phi)=\nabla_\theta L_\theta,\qquad
\nabla_\phi(L_\theta+L_\phi)=\nabla_\phi L_\phi.
\]

因此 `compute_loss()` 返回 `velocity + head`、一次反传，再由互不重叠的优化器分别更新，是 H11 的一种正确实现。两项必须先在更新前计算完成；不能在两次反传之间先更新一个网络。总和只用于执行两组半梯度更新，不能据此把 H 描述为普通联合 Gaussian NLL 优化。

隔离还必须延续到优化器操作：对全部参数一起做全局梯度范数裁剪，会让头梯度改变速度梯度的缩放。当前 workspace 对 theta/phi 分别裁剪、分别 AdamW 更新。`velocity_loss` 记录含对应辅助项的速度目标，`head_nll` 记录头目标；`bc_loss`/`train_loss` 是两项之和，不能直接当作跨有头/无头组的速度误差比较。

`rho=0` 的退化对象必须匹配架构、标签、辅助项及优化器设置：`theory` 档退化到同架构的普通 MSE；`mp1` 档退化到 `baseline_loss=mse` 的匹配对照（仍有 w=2 和 dispersive），不等于原 adaptive L2 基线。头仍独立训练，所以总损失数值不必等于无头基线，要求一致的是速度损失、theta 梯度及其实际更新。

审核时 `policy/mp1.py` 的原路径为：默认 `w=2`（135、138行），`v_hat=w*d+(1-w)*u_diag`（310行），标签为 `v_hat-h*dudt`（324行），MeanFlow损失用 residual adaptive L2（327、392–407行），再加 `0.5*dispersive_loss`（333行）。

应明确区分以下配置，不将它们混成一种算法：

| 配置 | 标签/其他损失 | 解释 |
|---|---|---|
| 原 MP1 baseline | 原 w=2、adaptive L2、dispersive系数0.5 | 保留实际原实现的性能对照 |
| theory 隔离基线/严格 H | 所有匹配组统一 w=1、普通平方回归、dispersive系数0；H组仅替换度量 | 最直接遵循 HTML H 及 PDF 修正公式 |
| MP1-compatible H | 所有匹配组统一原 w 与辅助损失，H以实际 `y_w` 监督头；统计主损失采用声明的平方/矩阵规则 | 可作为已授权的工程扩展，但不是未经改动的严格 H |

如果保留 w 混合，必须以实际标签 `y_w` 构造 e。对于固定 teacher 且 `v_hat=w*d+(1-w)*u_diag`，标签对 d 的线性系数为 wA，因此

\[
S_w=w^2ACA^\top,\qquad
M_w=S_w+(u-E[y_w\mid z])(u-E[y_w\mid z])^\top.
\]

不能把 w=1 目标训练的头直接当作 w=2 的统计头。dispersive项会继续影响主干，因此纯平方回归结论只适用于所隔离的回归项，不自动适用于整个训练系统。

adaptive L2 的权重由当前样本 `e^2` 生成，虽然detach，仍依赖隐藏标签。将它与 W 相乘会改变 H 的统计前提；若研究该组合，应另命名和消融。保留原 MP1 baseline，并增加匹配 w、辅助损失和普通MSE的 `none` 对照，才能区分“替换adaptive L2”和“新增协方差结构”的效果。

默认 profile 的选择应公开：强调逐式复现理论时默认 theory；强调原MP1底座上的工程兼容时可默认 mp1-compatible。无论默认哪种，都应将 profile、实际 y、主损失、w、辅助损失系数写入配置和checkpoint。不要把跨 profile 的差异全部归因于统计头。

## 5. 必须保留的解释边界与验收项

- SUREFlow 的整体实验不等于本方案仅训练加权已验证有效。H 的统计头、完整矩阵、有界权重和分离更新需要分别验证；B 是独立的对角联合NLL适配对照，不能叠入H后沿用同一解释。
- 旧文档记录的优化器兼容问题是待补的真实实现检查，不是数学矛盾；本次实现应在可用环境完成参数注册与实际更新检查。不得把显式手动梯度步冒充真实优化器已通过。
- 最小验收覆盖：实际MeanFlow标签/JVP；`M=S+bb^T`；正定下界；双向梯度隔离；同次更新前W；rho=0的student退化；r=t锚定；标量W恒等I；完整矩阵与对角的向量化次序；原MP1 baseline路径不被统计选项改变。
- 当所有有效开关关闭时，保持原预测接口、动作normalizer、动作切片及1NFE行为。若比较性能，须分开记录原MP1与matched-MSE，并固定相同推理预算。单元/代数测试通过不等于生成质量或机器人成功率提升。

