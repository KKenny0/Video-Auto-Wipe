<h1 align="center">Video Inpainting</h1>

**Changes compared to the original project**:
1. Partially refactored demo.py;
2. Added some performance optimization items;
3. README updated: environment dependencies, etc.

**Performance Comparison**:

| Test Data | Original | Optimized | Video Duration |
| :---: |:--------:|:---------:|:--------------:|
| chinese1.mp4 |   200s   |   125s    |      23s       |

---

If you are interested in AIGC application tools, you can learn a bit about it on [this blog](https://www.seeprettyface.com/).<br />
--------------------------------------------------------------------------------------------------<br /><br />

Erase the fixed-pattern content you don't want to see in your video. This project shares a model for subtitle removal and demonstrates the effectiveness of erasing content with easily recognizable patterns, such as subtitles, logos, and animated icons.<br /><br />

# Preview
## 1. Subtitle removal
![Image text](https://github.com/a312863063/Video-Auto-Wipe/blob/main/pics/de-text/detext_9_ko.JPG)<br/>
<p align="center"><a href='http://www.seeprettyface.com/mp4/video-inpainting/detext_06.mp4' target='_blank'>Check video</a></p><br/>
&emsp;&emsp;The function of the subtitle removal model is to automatically sense the location of subtitles in the video and erase them, and the sense subtitle method is to recognize subtitles with uniform styles.<br>&emsp;&emsp。<br/>
<br/><br/>

## 2. Logo removal
![Image text](https://github.com/a312863063/Video-Auto-Wipe/blob/main/pics/de-logo/delogo_4.JPG)<br/>
<p align="center"><a href='http://www.seeprettyface.com/mp4/video-inpainting/delogo_04.mp4' target='_blank'>Check video</a></p><br/>
&emsp;&emsp;The function of the icon tray model is that the model automatically collects the position of the icon in the video and then performs the tray. The method of collecting icons is that the pixel blocks that are still in the time domain are regarded as icons.<br/>
<br/><br/>

## 3. Dynamic logo removal
![Image text](https://github.com/a312863063/Video-Auto-Wipe/blob/main/pics/de-dynamic-logo/de-dynamic-logo_1.JPG)<br/>
<p align="center"><a href='http://www.seeprettyface.com/mp4/video-inpainting/de_dynamic_logo.mp4' target='_blank'>Check video</a></p><br/>
&emsp;&emsp;The function of the dynamic icon erasing model is that the model automatically senses the position of the dynamic icons in the video and then erases them. The method of sensing dynamic icons is that fixed pixel blocks that flicker or move dynamically in the time domain are regarded as dynamic icons.<br/>
<br/><br/>

# Installation
### 1.Environment Preparation
```commandline
opencv-python==4.12.0.88
matplotlib==3.10.3
numba==0.61.2
pysrt==1.1.2
tqdm==4.67.1
PyYAML==6.0.2
moviepy==2.1.2
```

# Usage
### 1.Get Started
- Download the pre-trained file and place it in the `pretrained_weight` folder.<br/>
- The pre-trained model download address: https://pan.baidu.com/s/1JN9-8Glw_ozOrSMgBIyHOw, Access code：px0s <br/> <br/>
- The download address of more input examples: https://pan.baidu.com/s/1_tzmvIoEQi3h_24-ieZJ_Q, Access code：cnqf <br/><br/>

&emsp;&emsp; Run```python demo.py```。

# Technical Details
## Training Data
### Background Data Preparation
1. A dataset of 2,709 movie clips was created based on more than 300 HD movies;<br/>
&emsp;&emsp;Download address：https://pan.baidu.com/s/1CIgJmFmx5iR2JfgAyjVaeg, Access code：xb7o <br/><br/>
2. A dataset of 864 TV show clips was created based on more than 40 HD TV shows.<br/>
&emsp;&emsp;Dwonload address：https://pan.baidu.com/s/1lJk6IIWlwxknAie0LlGYOg, Access code：9rd4 <br/><br/>

### Foreground Data Preparation
1. Subtitle removal： Utilizing the ImageDraw library to generate random styles and fonts, and simulate their variations.<br/>
2. Logo removal：Utilizing the ImageDraw library to generate random pixel areas, and simulate time-domain consistency (a fixed area in the video).<br/>
3. Dynamic logo removal：Utilizing PR software to simulate dynamic logo scenes.<br/><br/>
<br/>
### Training Process
&emsp;&emsp;Step 1. Temporal Domain Perception Training to specific tasks, that is, let the model sense the location of the foreground data needed to be erased in the video and perform erasure;<br/>
&emsp;&emsp;Step 2. Fusing the model to the erasure model, and then the model is trained end-to-end.<br/>
<br/><br/><br/>
