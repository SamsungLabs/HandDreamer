# HandDreamer: Zero-Shot Text to 3D Hand Model Generation using Corrective Hand Shape Guidance

<div align="center">

Green Rosh &nbsp;&nbsp;&nbsp; Prateek Kukreja &nbsp;&nbsp;&nbsp; Vishakha SR &nbsp;&nbsp;&nbsp; BH Pawan Prasad

Samsung R&D Institute India Bangalore

🎉 Accepted to CVPR 2026 🎉

</div>

---


<p align="middle">
<image src="HandDreamer.gif" width="80%">
<br>
<em>We propose HandDreamer: the first method for zero-shot 3D hand generation from text prompts. Our method generates highfidelity, geometrically accurate 3D hand models with diverse articulations from text prompts.</em>
</p>


## Abstract

The emergence of virtual reality has necessitated the generation of detailed and customizable 3D hand models for interaction in the virtual world. However, the current methods for 3D hand model generation are both expensive and cumbersome, offering very little customizability to the users. While recent advancements in zero-shot text-to-3D synthesis have enabled the  eneration of diverse and customizable 3D models using Score Distillation Sampling (SDS), they do not generalize very well to 3D hand model generation, resulting in unnatural hand structures, view-inconsistencies and loss of details. To address these limitations, we introduce HandDreamer, the first method for zero-shot 3D hand model generation from text prompts. Our findings suggest that view-inconsistencies in SDS is primarily caused due to the ambiguity in the probability landscape described by the text prompt, resulting in similar views converging to different modes of the distribution. This is particularly aggravated for hands due to the large variations in articulations and poses. To alleviate this, we propose to use MANO hand model based initialization and a hand skeleton guided diffusion process to provide a strong prior for the hand structure and to ensure view and pose consistency. Further, we propose a novel corrective hand shape guidance loss to ensure that all the views of the 3D hand model converges to view-consistent modes, without leading to geometric distortions. Extensive evaluations demonstrate the superiority of our method over the state-of-the-art methods, paving a new way forward in 3D hand model generation.

## Code

Coming soon...
