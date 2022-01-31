
# `Loba-NN:` &nbsp; <ins>*l*</ins>atent <ins>*o*</ins>scilatory <ins>*b*</ins>ayesian <ins>*a*</ins>nalysis

<p align="center">
  <kbd>
  <img src="https://github.com/SB-27182/loba-NN/blob/master/assets/examples/mod_loba-NN_readme/LatentOrtho_logo.png" width=185 height=170 />
  </kbd>
</p>

## `Goal: `
Mine relevant estimators from data where the data has been produced by an unknown oscillatory generator function.


## `Implementation: `
Loba-NN is an invertible-flow neural network that emphasizes the existence of hidden oscillatory manifolds in the data. Under the assumption of identifiability <sup>[*Sokol.*](https://arxiv.org/pdf/1401.7899.pdf)</sup>, it finds the conditionally-independent, latent probability manifold that partitions the behavior of the underlying generator function.
    
<br>

## `Technical-Details: `
 Compared to mainstream invertible-flow networks<sup>[*Baird.*](https://ieeexplore.ieee.org/document/1555983)[*Tabak.*](https://math.nyu.edu/~tabak/publications/Tabak-Turner.pdf)[*Dinh.*](https://arxiv.org/pdf/1410.8516.pdf)</sup>, loba-NN's primary differing attribute is it's latent density function:
<p align="center">
  <img src="Math/mod_latentPDF.png" />
</p>

&nbsp;<sub>*Note: It's not explicitly written in the above, but one should note that parameters conditioned on x<sub>3</sub> are in reality also conditioned on the weights of the relevant encoding neural network.<br>
 &nbsp; Note: I-naught is the zero-order Bessel equation.*</sub><br><br>

&nbsp; The *PDF* can readily be described as a conditional density function wherein the density of *Z* is Gaussian, conditioned on the mean being distributed by a von-Mises density. Further, the latent parameters are conditioned on the observed variable x<sub>3</sub>; this conditioning thus imparts identifiability for the ***disentangled***<sup>[*Tokui.*](https://arxiv.org/pdf/2108.13753.pdf)</sup> latent variables.<br>
&nbsp; Other than the latent density, the model uses affine coupling-layers<sup>[*Dinh.*](https://arxiv.org/pdf/1605.08803.pdf)</sup>, coupled batch-standardization<sup>[*Kingma*](https://arxiv.org/pdf/1807.03039.pdf)</sup>, and one conditional coupling layer<sup>[*Winkler.*](https://arxiv.org/pdf/1912.00042.pdf)</sup>.<br>
The overarching pattern is a standard invertible-flow network, *ie*: built to exploit the invariance property of maximum-likelihood estimators through maintaining bijectivity at each layered space-transform.

<br>

## `Strengths: ` 
As intended in the goal, loba-NN can mine conditionally varying estimators intrinsic to the generator function.

<br>

`Example:` &nbsp; Consider the data below. <br> It is drawn from a simulated dynamical system characterized by a ***subcritical Hopf-bifurcation***.<br>
We may note that the respective limit-cycle is longer on the y-axis; it emerges when the z-axis parameter (x<sub>3</sub>) is about some purportedly unknown critical value.<br>
Further complicating the data is that only in some instances does the respective saddle-node bifurcation (the limit-cycle) emerge; further, we recall that the nature of a subcritical Hopf-bifurcation is to impart ***hysteresis*** to the sampled data. <br>
Loba-NN is able to discover, not only the oscillator of the non-bifurcated phase-space, but also the latent behavior of the Hopf-bifurcation, as well as the ***lag*** given by the hysteresis of the dynamical system.

<p align="center">
  <img src="http://some_place.com/image.png" />
</p>
<br>


## `Mined Characteristics: ` 

<p align="center">
  <img src="http://some_place.com/image.png" />
</p>

&nbsp; Here we've used loba-NN to repeatedly generate 100 observations, conditioned on small regions of x<sub>3</sub>; &nbsp; &nbsp; *LL < x<sub>3</sub> < UL*<br>
We see that when the conditioning parameter is about 1.9, the system changes into an *oblong* shape. We would like to know if the manifold of the data is itself *oblong* shaped about this value of x<sub>3</sub>, or if there is perhaps something else going on.

<br>
<br>

<p align="center">
  <img src="http://some_place.com/image.png" />
</p>

&nbsp; Indeed, we can not simply assume the manifold is itself *oblong*. There is a huge spike in the variance when 1.8 < x<sub>3</sub> < 2.2 <br>
(*One may notice that shortly after the spike, the variance comes back down to a smoother value; I believe this is an indication of a hysteresis threshold involved in the system, however more analysis is forthcoming.*)<br>
Loba-NN is indicating it is much more *unsure* of the discovered oscillator when the conditioning parameter is around certain values.

<br>
<br>
<p align="center">
  <img src="http://some_place.com/image.png" />
  &nbsp;
  <img src="http://some_place.com/image.png" />
</p>
<br>
<p align="center">
  <img src="http://some_place.com/image.png" />
  &nbsp;
  <img src="http://some_place.com/image.png" />
</p>

To further investigate the behavior of the estimated generator function, we may artificially down-scale the conditional variance of one latent variable, and leave the other unchanged. This reveals that about the critical-value of x<sub>3</sub>, one latent variable supports the oscillator, while the other is in the process of learning the bifurcated limit-cycle. <br>

&nbsp; <sub> *Note: I use the phrase "process of learning" here because my 750ti GPU routinely has a hardware error around 2000 epochs. This is purely an issue on my end. Indeed, I'm unable to train most neural networks at all because of my GPU, despite it's CUDA compatibility. Although training any invertible-flow-network is extremely fast, a training session using the celebrity-faces<sup>[*1*](https://github.com/SB-27182)</sup> dataset crashes around 100 epochs.*</sub>


Fortunately, we may infer the trajectory of what's being learned by loba-NN.

<br>
<br>

## `Other Strengths: `

##### `Immunity to Mode Collapse:`
&nbsp; Unlike auto-regressive models, the existence of multiple clusters (and their modes) in the data does not cause a **logical error** *ie*: a mode-collapse. Although discrete structures such as multiple modes are not ideal for our purposes (more on that later), loba-NN, and all invertible-flow networks will happily bend hyper dimensional space to make sense of the observed probability density.


<br>

##### `Immunity to the "Curse of Dimensionality":`
&nbsp; Most clustering algorithms and auto-regressive models, *ie*: algorithms that optimize a euclidean measure in probability space, must face the realization that a 100-dimensional box has quite a lot of walls. This is to say, a box in 100 dimensions must have a combinatoric explosion worth of data points. Unlike euclidean measures, probabilistic measures of space are more abstract, there is no geometric kernel to them. Indeed, invertible-flow neural networks and their stochastic-mapping cousins, the VAE models<sup>[*Kingma.*](https://arxiv.org/pdf/1312.6114.pdf)</sup>, were intentionally built for hyper-dimensional data, *ie*: high resolution photos like celebrity-faces. 




<br>

##### `Direct Hypothesis Testing:`
&nbsp; Above, I've used loba-NN to transform the latent density distribution into the observed space. The nature of invertible neural networks is to enact this space transform in either direction. In the opposite direction, we may transform a set of observed values (like in an experimental sample) into the latent density. This means we may directly rate the accuracy of our statistical model based on how "extreme" the latent mappings, *ie:* **p-values** are. <br>
More interestingly, we may dive further into the latent representation and see exactly which independent latent variables agree/disagree with the new data. Unlike standard hypothesis testing encountered in a statistics course, we may analyze at a deeper level which aspects of the statistical model are good, and which may need updating.<br>
Note: This is analogous to anomaly detection using VAEs, save for the fact that our space transform is bijective and deterministic.


<br>

##### `Integration of the Learned Manifold for Practical Use:`
&nbsp; The insights loba-NN has discovered are easily incorporated into our existing quantitative/dynamical models. Indeed, the trained
neural network is itself, as per the notion of "identifiability", representing the conditional parameterization of the oscillatory system-of-ODEs.

`Biotech Example:` &nbsp; Suppose we have data obtained from a high-throughput, barcoding, transcriptional-NGS procedure. Using loba-NN as a simulation to guide our modifications, we may devise a transgenic cell-line that maintains a specific oscillatory homeostasis, wherein the cell-line expresses a set of transcriptional signals (*ie*: conditioning variables/ODE parameters), such that the cell-line is **NEVER**, or perhaps **ALWAYS**, undergoing the exampled Hopf bifurcation. <br>
Of course, there's more interesting phenotypes that can be developed beyond a simple boolean about a bifurcation; *eg*: functions of parameters in the stability-space of the dynamical system.

<br>
<br>


## `Weaknesses:`
&nbsp;The bijectivity of the probability-space transform performed by loba-NN and other invertible-flow networks allows one to specify structural qualities in the manifold of the data, however this bijectivity requirement means that discrete structures (wherein a probability-mapping/surjective-mapping must be used) such as clusters separated by large
euclidean distances are not suitable. Although Loba-NN will learn to place a low probability manifold in these spaces, it will still consider "something" to be there, when it should assume nothing is there.
<br>
This issue can be investigated by considering the differences of a ***stochastic mapping*** and a ***bijective mapping***. 
- ***Bijective mappings*** require that the space-transform from the input space to the output space, be a  continuous manifold; naturally this means that the domain and the co-domain must be equal in number of dimensions. In a nutshell, the only way loba-NN can interpret clusters is via "walking the distance" from cluster to cluster.
-  ***Stochastic mappings*** can map from a domain of any size, to a codomain of any size. This is because they learn a probabilistic density that maps between spaces. In simple words, any constraints placed on the latent density of a VAE, will be "jumped over" via a probability-mapping.

<br>

**Stochastic mapping NNs however, are very well equipped for hyper dimensional clustering <sup>[*Jiang.*](https://arxiv.org/abs/1611.05148)</sup>.**

**In short:** one should use a VAE to account for discrete structures in the data. Deeper insights can then be acquired by deploying an invertible-flow network, like loba-NN, on these partitioned continuous manifolds.




<br>
<br>

