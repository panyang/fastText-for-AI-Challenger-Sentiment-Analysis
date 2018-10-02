AI Challenger 2018 Sentiment Analysis Baseline with fastText
=========================================
功能描述
---
本项目主要基于AI Challenger官方[baseline](https://github.com/AIChallenger/AI_Challenger_2018/tree/master/Baselines/sentiment_analysis2018_baseline)修改了一个基于fastText的baseline，方便参赛者快速上手比赛，主要功能涵盖完成比赛的全流程，如数据读取、分词、特征提取、模型定义以及封装、
模型训练、模型验证、模型存储以及模型预测等。baseline仅是一个简单的参考，希望参赛者能够充分发挥自己的想象，构建在该任务上更加强大的模型。

开发环境
---
* 主要依赖工具包以及版本，详情见requirements.txt

项目结构
---
* src/config.py 项目配置信息模块，主要包括文件读取或存储路径信息
* src/util.py 数据处理模块，主要包括数据的读取以及处理等功能
* src/main_train.py 模型训练模块，模型训练流程包括 数据读取、分词、特征提取、模型训练、模型验证、模型存储等步骤
* src/main_predict.py 模型预测模块，模型预测流程包括 数据和模型的读取、分词、模型预测、预测结果存储等步骤


使用方法
---
* 准备 virtualenv -p python3 venv & source venv/bin/activate & pip install -r requirement.txt
* 配置 在config.py中配置好文件存储路径
* 训练 运行 python main_train.py -mn your_model_name 训练模型并保存，同时通过日志可以得到验证集的F1_score指标
* 预测 运行 python main_predict.py -mn your_model_name 通过加载上一步的模型，在测试集上做预测
* 更多详情请参考我的博客文章：http://www.52nlp.cn/?p=10537
