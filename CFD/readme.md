# Deep Temporal Compression
----------------------------------------------------

## Project layout

- All experimental code goes to 'experiments/*' (preferably in jupyter notebooks)
- 

## Ongoing Experiments
- SuperSloMo implementation for temporal interpolation (harsh)
    - Future milestones:
        - [ ] Adding Navier-Stokes loss
        - [ ] TempoGAN discriminator loss + feature-space loss
        - [ ] Add support for variable length interpolation
        - [ ] Expand to 3D data


## Datasets
- [Mantaflow](http://mantaflow.com/)
    - Widely used for many fluid dynamics experiments, allows exporting to numpy
    - Default dataset generator used in this project

## Relevant Links and Papers
- [Physics-Based-Deep-Learning](https://github.com/thunil/Physics-Based-Deep-Learning)
    - Best resource so far
- [TempoGAN: A Temporally Coherent, Volumetric GAN for Super-resolution Fluid Flow](https://arxiv.org/pdf/1801.09710.pdf)
- [Hidden Fluid Mechanics: A Navier-Stokes Informed Deep Learning Framework for Assimilating Flow Visualization Data](https://arxiv.org/abs/1808.04327)
    - [Related](http://www.dam.brown.edu/people/mraissi/research/1_physics_informed_neural_networks/)
- [Liquid Splash Modeling with Neural Networks](https://arxiv.org/pdf/1704.04456.pdf)
- [Deep Fluids: A Generative Network for Parameterized Fluid Simulations](https://arxiv.org/pdf/1806.02071.pdf)
    - [Github]https://github.com/byungsook/deep-fluids
- [Deep Learning Methods for Reynolds-Averaged Navier-Stokes Simulations of Airfoil Flows](https://arxiv.org/pdf/1810.08217.pdf)
    - https://github.com/thunil/Deep-Flow-Prediction

### Other resources
- [Differentiable Physics-informed Graph Networks](https://arxiv.org/abs/1902.02950)
- [Machine Learning for Computational Fluid and Solid Dynamics](http://www.cvent.com/events/machine-learning-for-computational-fluid-and-solid-dynamics/custom-17-b2f442696c984fc5bb84e1941befe281.aspx?dvce=1)
- https://www.youtube.com/watch?v=p45kQklIsd4&list=PLQY2H8rRoyvzoUYI26kHmKSJBedn3SQuB&index=12
- NERSC extreme weather nvidia keynote, WRF weather simulator
