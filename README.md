# RNN to Model Brain Maturation
The code is adapted from 
## Dependencies
The code is tested in Tensorflow 1.8.0, Python 2.7 and Python 3.6, and on MacOS 10.13 and Ubuntu 16.04.

Scikit-learn (http://scikit-learn.org/stable/) is necessary for many analyses.

The seaborn package (https://seaborn.pydata.org/) is needed to correctly
plot a few analysis results.

pic2video.py requires Opencv3. The pillow package and matplotlib package are needed to save the frames. 

PSTH_analysis.py requires statsmodels and pandas for ANNOVA analysis.


## Pretrained models
20 pretrained models and their auxillary data files for
analyses are provided on:
https://drive.google.com/drive/folders/1L8v-OZgYHVcKh1UKtCJl5QVlz8mkaRxr?usp=sharing

Download and unzip the file then copy /train_all folder to RNNPrefrontal/data.

## Reproducing results from the paper(Task representations in neural networks trained to perform many cognitive tasks 2019)
All analysis results from the paper can be reproduced from paper.py

Simply go to paper.py, set the model_dir to be the directory of your 
model files, uncomment the analyses you want to run, and run the file.

## Animated FTV and variance clustering changes
Create RNNPrefrontal/FTV_frame, RNNPrefrontal/FTV_video, RNNPrefrontal/variance_frame and RNNPrefrontal/variance_video folders

Run train_animation.py to generate the frames then run pic2video to produce the animated results.

### Example results

<p align="left">
	<img src="https://github.com/xinzhoucs/RNNPrefrontal/blob/master/Randodrd_ALLNEW256_fuse_onehot_input_FTV_20fps.gif" alt="Sample"  width="377" height="366">
	<p align="center">
		<em>FTV video</em>
	</p>
</p>

<p align="right">
	<img src="https://github.com/xinzhoucs/RNNPrefrontal/blob/master/Randodrd_ALLNEW256_fuse_onehot_input_variance_20fps.gif" alt="Sample"  width="324" height="204">
	<p align="center">
		<em>variance clustering video</em>
	</p>
</p>

## Additional notes
对lesion的说明:
在此处按需求注释/取消注释对应行
https://github.com/xinzhoucs/RNNPrefrontal/blob/9de86068b9968fb8dd93647744a93d5a085da470/network.py#L824

对ruleset的说明：
ruleset在此处：
https://github.com/xinzhoucs/RNNPrefrontal/blob/5dd6a3a48c3f48920dda5e71f8e126022eb427da/task.py#L9
