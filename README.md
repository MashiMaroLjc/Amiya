# :racehorse:  Amiya :rabbit2:
Amiya是一个提供开放域条件文本生成能力、具有少女人格的AI助手。


### 应用场景
Amiya不对标chatGPT及其他任意相似产品。Amiya的应用场景是在闲聊中应对条件文本生成的query,比如谈及天气时引用天气数据，在你抱怨某些事物时提供建议等。当两者遇到矛盾需要trade off时，Amiya的原则是对话过渡的平滑性、趣味性优先。


虽然amiya和GPT系列一样基于transformer结构的模型（目前基于T5结构）,但尺寸保证每个人都可以物理意义上拥有这个模型。不需要担心各种各样的问题无法使用、同时提供各位魔改、微调的机会。

### 声明

**本项目为玩具项目，模型训练中无任何安全考虑，使用过程中可能会生成让人不适的文本。使用者一旦使用本项目的模型权重，视作使用者为已经明确该情况和承担对应责任。因此，本项目对使用者滥用模型概不负责。**

# Update
- 2023.2.5 开放源码和提供checkpoint(第一版本代码为chatYuan的代码.https://huggingface.co/ClueAI/ChatYuan-large-v1)


# 依赖 

- python3.8+
- transformers 
- pytorch
- sentencepiece

# roadmap
-  [x] 开放代码+基础模型checkpoint 
-  [ ] 领域数据微调+人格注入
-  [ ] web app形式
-  [ ] query 生成能力、具备联网查询能力
-  [ ] 多种形式、类型的query生成、拓展技能边界
-  [ ] RLHF（有机会的话）

# 其他

纯粹是觉得阿米娅（amiya）这名字好听，适合做一个自己的人工助手，和明日方舟同名角色没啥关系。
