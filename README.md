# MultiTask Network

## Dependencies
The code is tested in Tensorflow 1.8.0, Python 2.7 and Python 3.6, and on MacOS 10.13 and Ubuntu 16.04.

Scikit-learn (http://scikit-learn.org/stable/) is necessary for many analyses.

The seaborn package (https://seaborn.pydata.org/) is needed to correctly
plot a few analysis results.

## Reproducing results from the paper
All analysis results from the paper can be reproduced from paper.py

Simply go to paper.py, set the model_dir to be the directory of your 
model files, uncomment the analyses you want to run, and run the file.

## Pretrained models
We provide 20 pretrained models and their auxillary data files for
analyses.
https://drive.google.com/drive/folders/1L8v-OZgYHVcKh1UKtCJl5QVlz8mkaRxr?usp=sharing
下载解压后将train_all文件夹放入data文件夹

主要使用：train.py,train_OZG.py,paper.py
对train和train_OZG.py：
```
parser.add_argument('--modeldir', type=str, default='data/OZG64')#add by yichen  <-改变此处路径来改变存储位置
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    hp = {'activation': 'softplus',
          'n_rnn': 64,
          'mix_rule': True,
          'l1_h': 0.,
          'use_separate_input': True} <-注意，此处默认是separate input，而paper.py中的lesion_unit默认fuse_input（改动见后文）
    for i in range(5): <- 控制不同的种子数
        train(args.modeldir,
            seed=i,
            hp=hp,
            ruleset='ozg', <-ruleset的说明见后文
            rule_trains=['overlap','zero_gap','gap'], 
            display_step=2) <-每隔多少步进行衡量（对简易任务建议设置较低数值）
```
对lesion的说明:
在此处按需求注释/取消注释对应行
https://github.com/xinzhoucs/RNNPrefrontal/blob/9de86068b9968fb8dd93647744a93d5a085da470/network.py#L824

对ruleset的说明：
ruleset在此处：
https://github.com/xinzhoucs/RNNPrefrontal/blob/5dd6a3a48c3f48920dda5e71f8e126022eb427da/task.py#L9

对paper.py的说明：
在paper.py中，注意：
https://github.com/xinzhoucs/RNNPrefrontal/blob/5dd6a3a48c3f48920dda5e71f8e126022eb427da/paper.py#L24
此处需对应到adult或young层文件夹(内部文件夹为012...）

训练使用train.py或train_OZG.py
分析结果使用paper.py
