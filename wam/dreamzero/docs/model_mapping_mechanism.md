# 模型映射机制原理说明

本文档详细阐述了在 `dreamzero` 项目中，配置文件中的 `architectures` 字段（例如 `["VLA"]`）如何映射到具体的 Python 模型类（如 `groot.vla.model.dreamzero.base_vla.VLA`），并解析其背后的实现原理。

## 1. 核心原理：Transformers 注册机制

本项目基于 Hugging Face `transformers` 库构建。该库提供了一套自动模型加载机制，核心依赖于两个注册表：
1.  **Config 注册表 (`AutoConfig`)**：将字符串标识符（`model_type`）映射到配置类。
2.  **Model 注册表 (`AutoModel`)**：将配置类映射到具体的模型实现类。

当用户调用 `AutoModel.from_pretrained(path)` 时，内部流程如下：
1.  读取路径下的 `config.json`。
2.  提取 `architectures` 列表中的第一个元素（类名）或 `model_type` 字段。
3.  在注册表中查找对应的类。
4.  实例化该类别并加载权重。

## 2. 代码实现分析

在 `groot/vla/model/dreamzero/base_vla.py` 文件中，我们定义了配置类和模型类，并显式地进行了注册。

### 2.1 定义配置类 (VLAConfig)

首先，定义继承自 `PretrainedConfig` 的配置类。关键在于设置 `model_type` 属性，这是映射的“键”。