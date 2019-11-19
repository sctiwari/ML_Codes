<center> <h1> Deep Learning Based Spatio-temporal Interpolation in Fluid Dynamics </h1> </center>
<h3><center>
Xiangyu Gao, Jun Luo, Harsh Mohan, Subodh C. Tiwari, Ziqi Zeng</center>
</h3>
<h4><center>
 Team: Mom Tells me Professor Doesn't like Long name <br><br>
 University of Southern California
</center></h4>

## Introduction

<div style="text-align: justify">

Computational Fluid dynamics is a basic building block for designing modern engineering applications such as aerodynamics systems, weather simulations, industrial pipelines. An archetypical fluid simulation involves solving complex Navier-Stokes equations, which has been designated as one of the 7 millennium problems.<a href="#ref1">[1]</a> A numerical solution of fluid-dynamics is computationally expensive and incur large storage cost. In this post, we will discuss a novel way to compress data using Deep learning spatial and temporal interpolation, contrary to common method of storing data such as temporal coarsening. Further, we can use temporal interpolation to accelerate comutational fluid dynamics.
</div><br>

<center><table width=50%>
<tr>
<td> <img src="https://i.imgur.com/UZsnRZt.gif" width="100%"> </td>
<td>
<img src="https://imgur.com/aC4FPMD.gif" width="100%">
</td>
</tr>
</table></center>

<center><i> <b> Illustration of Fluid dynamics data.</b></i></center>



<div style="text-align: justify">
Our method of temporal coarsening our data takes inspiration from a commonly known method for video interpolation to generate slow motion video. Several techniques have been proposed using single and double frame.<a href="#ref2">[2-5]</a> However,  such method can be applied to scientific data where, we can save temporally coarse data and a trained network. Further, a trained network can generate the deleted frame at later time without recomputing. To furhter compress our data spatially, we employ super resolution technique, which has only been used on RGB image so far. <a href="#ref6">[6-8]</a> To best of our knowledge, no study has been performed so far using deep learning method to compress fluid dynamics data. 
</div><br>

<center><img src="https://i.imgur.com/5IeulmC.png" width="50%"> </center>
<center><i> <b>An End-to-end pipeline to re-generate fine-grain data spatially and temporally.</i></center></b>

## Dataset 
<div style="text-align: justify">
Deep-learning models require large, diverse datasets across several physical scenarios. Currently, the interpolation is performed only on the density field. We write our own simulations using an open source solver, MantaFlow. We generate 40,000+ data points across two types of scenes - Plume and Karman Vortex Street.
</div>

<b> insert movie of two problems </b>

## Loss Function for Fluid Dynamics 

## Network Architechure

### Temporal Interpolation
<div style="text-align: justify">

Our first task involves temporal interpolation. We first use fluid state $S$ to represent a spatial matrix consisting of several fluid properties including the velocity, density and pressure. Now, given $S_0$ and $S_N$ at timesteps $t_0$ and $t_N$ respectively, our goal is to design a function $f_t$ that approximates the fluid state $S_t$ at some time $t$ such that $t \in [t_0,t_N]$. <br>
</div>

$$
f_t(S_0, S_N, t_0, t_N, t ) \cong S_t
$$

<div style="text-align: justify">

Our current model is inspired from a state-of-the-art arbitrary video frame interpolation, Super SloMo <a href="#ref2">[2]</a>. We introduced two major modification in the network as follows:
<ul>

<li> We avoid calculating bi-directional optical flow and explicitly use second submodule for end-to-end training. This model is implemented as a U-Net <a href="#ref5">[5]</a> style encoder-decoder module. </li>
<li>We also avoid using batch normalization since we model real world quantities that can be arbitrarily large or small, and can vastly differ across different instances in a batch because of different physics. </li>
</ul>
</div>
<center><img src="https://i.imgur.com/wIHsgiI.png" width="50%"> </center>


### Sptial Interpolation
<div style="text-align: justify">

To compress the physical simlulation spatially, we can conduct the simulation with low spatial resolution of each frame $S_l$, and use spatial deep learning models $F_s$ to enhance the frame to a higher resolution $S$. i.e.


$$
F_s(S_l) \cong S
$$


We implemented and compared two deep learning architectures inspired by Super Resolution Convolutional Neural Network (**SRCNN**) <a href="#ref7">[7]</a> -- the vanilla SRCNN and our improved modified SRCNN. A detailed comparison of the two models are shown below.

<center><img src="https://i.imgur.com/GDlZ6GN.png" width=50% ></center>

<center><i><b>Architecture of SRCNN and Modified SRCNN</b></i></center><br>





We also implemented a more recent architecture, Super Resolution Generative Adversarial Network (**SRGAN**) <a href="#ref8">[8]</a>, which employed the block layout, small kernels, and Parametric-ReLU as the activation function.<br>
Generator uses MSE as loss function, which compares the generated data with ground truth. Discriminator uses BCE as loss function, which compares the ground truth data with True label and generated data with False label.
</div>




<center><img src="https://i.imgur.com/FZ9ZJD2.png"></center>
<center><i><b>SRGAN Generator</b></i></center><br>

<center><img src="https://i.imgur.com/LZSBP4u.png"></center>
<center><i><b>SRGAN Discriminator</b></i></center>

## Experiments

### Calculation of Error and Mean Relative Error
<div style="text-align: justify">

We used the Mean Squared Error (MSE) as the loss for training the three spatial interpolation models and for test set. We also computed the mean relative error for test set by

$$
\sum_{i,j} \frac{y_{t_{i,j}}-y_{p_{i,j}}}{y_t}\times100\%
$$

for each frame, where $y_{t_{i,j}}$ and $y_{p_{i,j}}$ mean the true and predicted value for the i-th row, j-th column grid cell.
</div>

### Temporal Interpolation
<div style="text-align: justify">

Based on our experimentation, our model is able to reconstruct the simulations fairly well. Besides this, our model also proved to be 30-60 times faster depending on the complexity of the simulation. A summary of our results and some reconstructions can be seen below.
</div><br>


<table align="center" width=50%>
<tr>
<td width="220px" align="center"><b>Model/Reconstruction</b></th>
<td width="220px" align="center"><b>Mean absolute error</b></th>
<td width="220px" align="center"><b>Mean relative error</b></th>
</tr>
<tr>
<td align="center">Base - 1 frame window</td>
<td align="center"><b>0.00080</b></td>
<td align="center"><b>3.37%</b></td>
</tr>
<tr>
<td align="center">Base - 2 frames window</td>
<td align="center">0.0018</td>
<td align="center">7.79%</td>
</tr>
<tr>
<td align="center">Base - 3 frames window</td>
<td align="center">0.0030</td>
<td align="center">12.68%</td>
</tr>
</table>

<table width="50%">
<tr> 
<td width="220px" align="center"> <b> Original </b> </td>
<td width="220px" align="center"> <b> Reconstructed</b></td>
<td width="220px" align="center"> <b> Error </b> </td>
</tr>

<tr>
<img src="https://imgur.com/Fi0AMZG.gif">
</tr>
<tr>
<img src="https://imgur.com/RCvyTXR.gif">
</tr>
</table>






### Sptial Interpolation

<div style="text-align: justify">

The reconstruction results of all three spatial models are reasonable. SRGAN model turns out to have the smallest reconstruction loss among the three. Our improved modified SRCNN comes second. A summary of the results is shown below. Note that although the results below are shown in RGB format, the actual and predicted value our models predict are of actual physical meaning.
</div><br>

<table align="center">
<tr>
<td width="220px" align="center"><b>Model</b></th>
<td width="220px" align="center"><b>Mean absolute error</b></th>
<td width="220px" align="center"><b>Mean relative error</b></th>
</tr>
<tr>
<td align="center">SRCNN</td>
<td align="center">0.0018</td>
<td align="center">15.23%</td>
</tr>
<tr>
<td align="center">Modified SRCNN</td>
<td align="center">0.0014</td>
<td align="center">12.47%</td>
</tr>
<tr>
<td align="center">SRGAN</td>
<td align="center"><b>0.0006</b></td>
<td align="center"><b>11.52%</b></td>
</tr>
</table>
<center><i><b>Error of Different Models in Spatial Interpolation</b></i></center><br>


<img src="https://i.imgur.com/PDxqLIl.png">



<div style="text-align: justify">

</div><br>

<div style="text-align: justify">
We randomly generated some outputs with randomly generated inputs. As shown in the figures above, the reconstructed results of SRCNN with size 128x128 from the low-resolution frame with size 32x32 is more blurry than the results of SRGAN. SRCNN produces outcomes with more granular sensation, while SR-GAN has more artifacts.
</div>


## Future work
- Experiment with continuous filter convolutional neural networks.
- Improve Navier-Stokes constraints loss term implementation.
- Experiment and train on several other physical phenomena.

## Conclusion
We have created a novel way to compress fluid-dynamic data and accelerate fluid dynamics simulations using spatial and temporal interpolation. 


## References
<span id="ref1" name="ref1">1</span>- A. M. Jaffe. The Millennium Grand Challenge in Mathematics. Notices of the AMS 53(6):652-660, 2000
<span id="ref2" name="ref2">2</span>- H. Jiang, D. Sun, V. Jampani, M.-H. Yang, E. Learned-Miller, J. Kautz. Super SloMo: High Quality Estimation of Multiple Intermediate Frames for Video Interpolation. CVPR, 2018
<span id="ref3" name="ref3">3</span>- Z. Liu, R. Yeh, X. Tang, Y. Liu, and A. Agarwala. Video frame synthesis using deep voxel flow. In ICCV, 2017.
<span id="ref4" name="ref4">4</span>- S. Niklaus, L. Mai, and F. Liu. Video frame interpolation via adaptive convolution. In CVPR, 2017
<span id="ref5" name="ref5">5</span>- S. Niklaus, L. Mai, and F. Liu. Video frame interpolation via adaptive separable convolution. In ICCV, 2017.
<span id="ref6" name="ref6">6</span>- O. Ronneberger, P. Fischer, T. Brox. U-Net: Convolutional Networks for Biomedical Image Segmentation, arXiv:1505.04597v1.
<span id="ref7" name="ref7">7</span>- C. Dong, C. C. Loy, K. He, X. Tang. Image Super-Resolution Using Deep Convolutional Networks, arxiv:1501.00092
<span id="ref8" name="ref8">8</span>- C. Ledig, L. Theis, F. Huszar, J. Caballero, A. Cunningham, A. Acosta, A. Aitken, A. Tejani, J. Totz, Z. Wang, Wenzhe Shi. Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network, arXiv:1609.04802v5







## Appendix and FAQ

:::info
**Find this document incomplete?** Leave a comment!
:::

###### tags: `Templates` `Documentation`

