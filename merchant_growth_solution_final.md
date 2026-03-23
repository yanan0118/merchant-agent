# Merchant Growth Copilot — Product Solution (Final)

## 1. 产品定位（Positioning）
面向客户经理（Account Manager）的**AI经营决策系统**，将原本由少数运营专家掌握的经营分析与增长策略能力产品化、规模化。

> 从“关系驱动” → “洞察驱动”的工作方式转变

---

## 2. 背景与问题（Background）
当前客户经理在推进商家增长时面临：
- 数据获取门槛高（BI复杂难用）
- 缺乏分析框架（不会看、不会用）
- 依赖运营人力（仅覆盖头部商家）

本质问题：
> 经营决策能力没有被系统化与规模化

---

## 3. 目标（Goals）
1. 快速判断商家经营健康度  
2. 自动识别增长短板  
3. 生成可执行行动建议  

---

## 4. 核心方法论（Core Framework）

### Merchant Growth Diagnosis Framework
```
Health → Driver → Gap → Action
```

---

## 5. Growth Health Model（增长健康度）

### 核心逻辑
结合**时间维度 + 横向维度**的分布判断：

- Historical Percentile（过去30天）
- Peer Percentile（同品类）

### 分类规则
- Healthy：均正常  
- Warning：任一偏高  
- Risk：历史Top10% 且 同行Top30%

### 优势
- 避免拍脑袋阈值  
- 兼顾趋势与竞争  

---

## 6. Driver Decomposition（驱动拆解）

四大增长驱动：

1. 流量（曝光、转化率）  
2. 基础经营（图片、营业时长）  
3. 用户体验（出餐、取消率）  
4. 供给质量（SKU、折扣）  

---

## 7. Gap Model（短板识别）

### 主逻辑：Relative（竞争导向）

```
Percentile < 30% → Gap
```

### 辅助：Absolute校准

```
Relative差 + Absolute不达标 → Strong Gap
```

### 输出
- Strong Gap / Gap / Normal / Strong

---

## 8. Action Engine（行动决策）

### 核心公式
```
Action Score = Gap Severity × Driver Weight
```

### 特点
- 展示顺序：固定（认知稳定）
- 决策逻辑：动态（智能优化）

---

## 9. 系统架构（System Design）

```
User Query
→ SQL Generation
→ Data Fetch
→ Diagnosis Model
→ LLM Explanation
```

---

## 10. 示例输出（Example）

输入：商家A

输出：
- 健康度：Risk  
- 短板：曝光低、图片覆盖低  
- 建议：
  - 延长营业时间（High）
  - 优化图片（Medium）

---

## 11. 组织价值（Org Impact）

### 1. 能力下沉
运营专家能力 → 系统能力

### 2. 规模化
100%商家覆盖

### 3. 行为改变
客户经理：
- 从“关系沟通” → “数据说服”

---

## 12. Roadmap

- MVP：单商家诊断（Notebook）
- V2：CLI / API
- V3：批量 + 推荐系统
- V4：策略优化闭环

---

## 13. 总结（Summary）

本产品不是一个工具，而是：

> 一个将经营增长能力系统化、AI化的决策引擎
