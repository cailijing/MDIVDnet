&nbsp;

<div align="center">

<h2> Exploring Video Denoising in Thermal Infrared Imaging: Physics-inspired Noise Generator, Dataset and Model </h2> 

Lijing Cai, 
Xiangyu Dong,
Kailai Zhou and
[Xun Cao](https://scholar.google.com/citations?user=8hZIngIAAAAJ&hl=zh-CN&oi=ao)

*[Lab for Computational Imaging Technology & Engineering](https://cite.nju.edu.cn/), Nanjing University, China*



</div>

&nbsp;

### <summary><b>Introduction</b></summary>
Perception in the thermal infrared significantly enhances the capabilities of machine vision. Nonetheless, noise in imaging systems is one of the factors that hampers the large-scale application of equipment. We endeavor on a rarely explored task-thermal infrared video denoising and hope that like-minded individuals will join us in advancing this field.

This is the official repository for physics-inspired infrared video-level noise generator, TIVID, and MDIVDnet. Please feel free to raise any questions about them at any time! If you find this repo useful, please give it a star ⭐ and consider citing our paper. Thank you.

&nbsp;
### <summary><b>Contributions</b></summary>

<details close>
<summary><b>Physics-inspired noise generator</b></summary>

![test](/fig/generator.png)

</details>

<details close>
<summary><b>Thermal Infrared Video Denoising Dataset (TIVID)</b></summary>

![results2](/fig/TIVID.png)

</details>

<details close>
<summary><b>Multi-Domain Infrared Video Denoising Network (MDIVDnet)</b></summary>

![results3](/fig/MDIVDnet.png)

</details>

&nbsp;

### <summary><b>Performance</b></summary>

<details close>
<summary><b>Comparison with video denoising methods</b></summary>

![results4](/fig/video_denoising_methods.png)

</details>

<details close>
<summary><b>Comparison with thermal infrared denoising methods</b></summary>

![results5](/fig/infrared_denoising_methods.png)

</details>

<details close>
<summary><b>Temporal Consistency</b></summary>

![results6](/fig/Temporal_Consistency.png)

</details>

<details close>
<summary><b>Comparison Results for Real-World Noise</b></summary>

![results7](/fig/real_world_noise.png)

</details>





&nbsp;

## 1. Create Environment:

We recommend using [Conda](https://docs.conda.io/en/latest/miniconda.html) to set up an environment.

``` sh
# Create environment
conda create -n MDIVDnet python=3.11
conda activate MDIVDnet

# Install pytorch 
conda install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=12.1 -c pytorch -c nvidia

# Install open-mmlab packages
pip install -U openmim
mim install mmcv==2.1.0
mim install mmengine==0.10.4
mim install mmagic==1.2.0 

NOTE：If there are any issues with installing the open-mmlab packages, please refer to
 · https://github.com/open-mmlab/mmcv
 · https://github.com/open-mmlab/mmagic

# Install other packages
pip install -r requirements.txt
```



&nbsp;

## 2. Prepare Dataset:

Download our processed datasets from [Google drive](https://drive.google.com/file/d/1ytcmaj_Niv_EVMH10EKLTiUhODIlb2r9/view?usp=sharing),  [Baidu disk](https://pan.baidu.com/s/13rxgKvVXvZo3L2O6KvOmQg?pwd=13zp), or [NJU box](https://box.nju.edu.cn/f/17444f0a06ee4a9ba19c/). Then put the downloaded datasets into the folder `data/` as

```sh
  |--data
      |--test_MP4
      |--test_PNG
      |--train_MP4
      |--train_PNG
```

&nbsp;

## 3. Testing:

```sh
python test.py 
```

&nbsp;

## 4. Training:


```sh
python train.py 
```


&nbsp;

## 5. Citation
If this repo helps you, please consider citing our works:


```sh
@article{Cai2024,
  title = {Exploring Video Denoising in Thermal Infrared Imaging: Physics-Inspired Noise Generator,  Dataset,  and Model},
  volume = {33},
  ISSN = {1941-0042},
  url = {https://ieeexplore.ieee.org/document/10507231},
  DOI = {10.1109/tip.2024.3390404},
  journal = {IEEE Transactions on Image Processing},
  publisher = {Institute of Electrical and Electronics Engineers (IEEE)},
  author = {Cai,  Lijing and Dong,  Xiangyu and Zhou,  Kailai and Cao,  Xun},
  year = {2024},
  pages = {3839–3854}
}
```
