# HandDreamer: Zero-Shot Text to 3D Hand Model Generation using Corrective Hand Shape Guidance

<div align="center">

[Green Rosh](https://www.linkedin.com/in/green-rosh-k-s) &nbsp;&nbsp;&nbsp; [Prateek Kukreja](https://www.linkedin.com/in/prateek-kukreja) &nbsp;&nbsp;&nbsp; [Vishakha SR](https://www.linkedin.com/in/vishakha-sr) &nbsp;&nbsp;&nbsp; [Pawan Prasad B H](https://www.linkedin.com/in/pawaniitm)

Samsung Research Institute India - Bangalore

🎉 Accepted to CVPR 2026 🎉

[Paper](https://arxiv.org/abs/2604.04425)


</div>

---


<p align="middle">
<image src="HandDreamer.gif" width="80%">
<br>
<em>We propose HandDreamer: the first method for zero-shot 3D hand generation from text prompts. Our method generates highfidelity, geometrically accurate 3D hand models with diverse articulations from text prompts.</em>
</p>


## Abstract

The emergence of virtual reality has necessitated the generation of detailed and customizable 3D hand models for interaction in the virtual world. However, the current methods for 3D hand model generation are both expensive and cumbersome, offering very little customizability to the users. While recent advancements in zero-shot text-to-3D synthesis have enabled the  eneration of diverse and customizable 3D models using Score Distillation Sampling (SDS), they do not generalize very well to 3D hand model generation, resulting in unnatural hand structures, view-inconsistencies and loss of details. To address these limitations, we introduce HandDreamer, the first method for zero-shot 3D hand model generation from text prompts. Our findings suggest that view-inconsistencies in SDS is primarily caused due to the ambiguity in the probability landscape described by the text prompt, resulting in similar views converging to different modes of the distribution. This is particularly aggravated for hands due to the large variations in articulations and poses. To alleviate this, we propose to use MANO hand model based initialization and a hand skeleton guided diffusion process to provide a strong prior for the hand structure and to ensure view and pose consistency. Further, we propose a novel corrective hand shape guidance loss to ensure that all the views of the 3D hand model converges to view-consistent modes, without leading to geometric distortions. Extensive evaluations demonstrate the superiority of our method over the state-of-the-art methods, paving a new way forward in 3D hand model generation.

## Hardware requirements

We recommend a cuda compatible GPU with atleast 40 GB of memory. Our solution is tested on an NVIDIA RTX A6000 with 48 GB of memory.

## Installation

### Set up ThreeStudio

This project is built on top of [Threestudio](https://github.com/threestudio-project/threestudio). You can skip this part if threestudio is already set up on your system. We recommend installation using a virtual environment.

```
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install ninja
pip install -r requirements.txt
``` 

More details on this step can be obtained [here](https://github.com/threestudio-project/threestudio).


### Set up MANO

Install the necessary smplx packages

```
pip install smplx
pip install chumpy
```

MANO weights needed to be downloaded for this implementation. Follow the instructions in [mano](https://github.com/vchoutas/smplx#downloading-the-model) for license agreements and model downloads. Once the models are downloaded, keep them in the following folder structure:
```
HandDreamer
|--threestudio/                         
|  |--models/
|  |  |--smpl_models/
|  |  |  |--mano/
|  |  |  |  |--MANO_LEFT.pkl
|  |  |  |  |--MANO_RIGHT.pkl
```

## Set the parameters

The parameters used by HandDreamer is provided at 

```
configs/handdreamer_config.yaml
```

The output directory needs to be set using:

```
name: <folder_name>   #name of the folder for current experiment
exp_root_dir: <full path to outputs folder>    #Recommended to provide absolute path instead of relative path
```

The MANO root folder needs to be set using:

```
data:
  mano_root: </full/path/models/smpl_models>
```

Number of training steps can be set using:

```
guidance:
  trainer_max_steps: 24000
trainer:
  max_steps: 24000

```

Default value is 24000 for high quality 3D hand generation. Faster 3D generation  can be achieved by setting this value lower (Eg: A value of 10000 converges in ~45 minutes). Please ensure to set both the parameters to the same value

## Generating 3D hands from text

We have provided a few sample prompts. Please run the following command to generate neural representation (NeRF) of 3d hands from text prompts

```
sh sample_prompts.sh 
```

The results will be saved to:

```
save_dir = <exp_root_dir>\<name>\<text_prompt>
```
## Export to mesh

The generated hand nerf can be exported to a 3d mesh using marching cubes algorithms as follows:

```
python launch.py --config <save_dir>/configs/parsed.yaml --export --gpu 0 resume=<save_dir>/ckpts/last.ckpt system.exporter_type=mesh-exporter system.geometry.isosurface_method=mc-cpu system.geometry.isosurface_resolution=256
```

## Hand Articulation

The default parameters generate 3D hand in default MANO pose. A different pose can be obtained by changing the initial MANO pose.

## Acknowledgements

We are grateful for the following excellent repository, from which our project benefits:

[Threestudio](https://github.com/threestudio-project/threestudio)

## Citation

If you find this repository useful for your work, kindly consider citing it as follows:

```
@article{rosh2026handdreamer,
  title={HandDreamer: Zero-Shot Text to 3D Hand Model Generation using Corrective Hand Shape Guidance},
  author={Rosh, Green and Kukreja, Prateek and SR, Vishakha and others},
  journal={arXiv preprint arXiv:2604.04425},
  year={2026}
}
```
