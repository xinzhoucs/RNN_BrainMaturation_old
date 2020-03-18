# RNN to Model Brain Maturation
This code is adapted from the <a href="https://github.com/gyyang/multitask">Multitask</a> and <a href="https://github.com/nmasse/Short-term-plasticity-RNN">Short-term-plasticity-RNN</a>.

We train RNN to learn working memory task (ODR and ODRD) and anti-saccade task (Overlap, Zero-gap, and Gap).

<p align="center">
	<img src="https://github.com/xinzhoucs/RNNPrefrontal/blob/master/example/Tasks.jpg"  width="522" height="188">
</p>

## Dependencies
The code is tested in Tensorflow 1.8.0, Python 2.7 and Python 3.6, and on MacOS 10.13 and Ubuntu 16.04.

Scikit-learn (http://scikit-learn.org/stable/) is necessary for many analyses.

The seaborn package (https://seaborn.pydata.org/) is needed to correctly
plot a few analysis results.

pic2video.py requires Opencv3. The pillow package and matplotlib package are needed to save the frames. 

PSTH_analysis.py requires statsmodels and pandas for ANNOVA analysis.

## Animated Fractional Task Variance (FTV) and Clustering for Task Representation during Maturation

Run train_animation.py to generate the frames then pic2video.py to produce the animated results.

### Example results

<p align="center">
	<img src="https://github.com/xinzhoucs/RNNPrefrontal/blob/master/example/Randodrd_ALLNEW256_fuse_onehot_input_FTV_20fps.gif" alt="Sample"  width="377" height="366">
	<p align="center">
		<em>FTV video</em>
	</p>
</p>

<p align="center">
	<img src="https://github.com/xinzhoucs/RNNPrefrontal/blob/master/example/Randodrd_ALLNEW256_fuse_onehot_input_variance_20fps.gif" alt="Sample"  width="324" height="204">
	<p align="center">
		<em>Clustering for Task Representation video</em>
	</p>
</p>

## PSTH analysis
Run train_PSTH.py then PSTH_analysis.py. The analysis results will be saved in RNNPrefrontal/figure/figure_data

### Example results

<p align="center">
	<img src="https://github.com/xinzhoucs/RNNPrefrontal/blob/master/example/PSTH_bygrowth_520960to628480.png"  width="402" height="282">
	<p align="center">
		<em>PSTH_bygrowth_520960to628480</em>
	</p>
</p>

## Additional notes
For ruleset：
Rulesets are listed here：
https://github.com/xinzhoucs/RNNPrefrontal/blob/64d3af827df8899f1759cef0fecfe3b92ca73c68/task.py#L8
