# YOLO26n smoke-test 验证异常排查报告 (EXP-001B Task 3)

Date (UTC): 2026-11-03 · Scope: 短程诊断，无新训练 · Data: `yolo26_val_diagnosis.json`

## 现象

模型选择 smoke test（合成数据，2 epoch）中 yolo26n 训练后验证
P=0.031 / R=1.0 / mAP50=0.037 —— 远低于同条件 yolo11n (0.70) 与
yolo26s (0.72)。需要确认是模型缺陷、配置错误还是测量假象。

## 复核项

| 复核项 | 结果 |
|---|---|
| 预训练权重 | `weights/yolo26n.pt` 5.54MB，SHA-256 `9b09cc8b…f4fef`，来自官方 ultralytics assets v8.4.0；训练中 `Transferred 696/708 items`（未迁移项=类别数不匹配的 cls 头卷积） |
| 类别映射 | 训练时 `Overriding model.yaml nc=80 with nc=1`；`cls_remap=True`；合成数据 1 类 `weed`；训练后 eval 输出 `[1,5,8400]`（4+1）确认 nc=1 生效 |
| 训练配置 | 三模型完全一致：epochs=2, imgsz=640, batch=8, seed=42, workers=0, deterministic=True, optimizer=auto→AdamW(0.002), amp=True |
| 双头结构 | Detect 头含 `one2one_cv2/cv3`（one2one 子头）；`reg_max=1`（DFL-free）；eval 前向返回 `{one2many:{...}, one2one:{...}}` |

## 源码定位（ultralytics 8.4.182）

- `engine/validator.py:167-168`：`model.end2end = self.args.nms is False` → **默认验证使用 one2many 头 + NMS**；`nms=False` 才切到 one2one（NMS-free）头。
- `nn/modules/head.py:196`：训练模式下 **one2one 头输入被 detach**（`x_detach = [xi.detach() for xi in x]`，注释：detach keeps one2one out of the backbone）→ 主干仅由 one2many 监督；one2one 只学自己的头卷积。
- `nn/modules/head.py:201`：推理按 `end2end` 标志在 `preds["one2one"/"one2many"]` 间选择。

## 关键实验（对已训 best.pt 各跑一遍 val，秒级）

| 验证路径 | P | R | mAP50 | mAP50:95 |
|---|---|---|---|---|
| yolo26n 默认（one2many + NMS） | 0.024 | 1.000 | 0.037 | 0.027 |
| yolo26n `nms=False`（one2one 头，无 NMS） | 0.500 | 0.513 | **0.432** | 0.337 |
| yolo11n 单头 | 0.746 | 0.601 | 0.696 | 0.474 |

## 解释

1. **并非模型缺陷**：one2one（部署用 NMS-free）头在同一次训练后 mAP50=0.432，
   与 yolo11n 同量级。被报告的 0.037 来自默认验证所选的 **one2many** 头。
2. **one2many 低分机制**：val 用 conf≈0.001 计分；2 epoch 下 one2many 头的
   分类分支（nc=80→1 重新初始化，且其还承担对 one2one 的主干监督职责）
   分数远未校准，输出海量低分候选 → R=1.0 但 P≈0.02，mAP 崩。one2one 头
   输出受 max_det 天然约束，无此问题。
3. **附带结论**：训练期每 epoch 打印的验证指标（0.037）系统性低估
   YOLO26 的部署路径性能——评估 YOLO26 时必须同时记录两种头路径，
   且论文/报告中应注明所用路径；**这再次证明 2-epoch 指标不能作排名依据**。
4. Progressive Loss 随训练将监督权重移向 one2one（推理头），2 epoch 时该
   转移几乎未发生，one2many 未充分训练亦符合预期。

## 对协议的影响（建议，不执行）

- EXP-002 基线验证时，YOLO26 模型需同时记录 `val(one2many+NMS)` 与
  `val(nms=False)`；最终以部署路径（one2one NMS-free）为准并全文注明。
- 模块 C 的特征一致性作用于 neck 特征，与头路径无关，不受影响。

## 状态：**已解释（无缺陷）**
