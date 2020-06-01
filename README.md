# Simulating choice model of graph edge formation.

This repository contains partial code base to the following paper:

Scaling choice models of relational social data - Jan Overgoor, George Pakapol Supaniratisai, Johan Ugander. (KDD, 2020)

The code base contains the synthetic graph experiments portion of the paper which includes the following routine:

- Simulating graph edge formation under both regular conditional logit (single-mode multinomial) choice model and mixed mode multinomial choice model
- Feature extraction under different hyperparameters:
  * Sampling methods
  * Candidates subsampling size
  * Events subsampling size
- Choice model fitting (single and de-mixed).

In this part, we used the following versions of external python libraries:

- `numpy=1.18.1`
- `scipy=1.2.0`
- `torch=0.4.0` (to accelerate the optimizing routine)

To generate data for figure 3 for conditional logit on synthetic graph, run:
    ```
    mkdir MNL-graph
    python3 generate_MNL_graph.py
    python3 MNL_model_experiment.py
    ```
    
* Running the code will generate graph in the folder `MNL-graph` and run all experiments in the one-mode conditional logit section. 
* In `MNL_model_experiment.py` there are 3 commands: `fig_3a_3b()`, `fig_3c()`, and `fig_3d()`. which can be imported imported as a module function to generate  `synthetic-vary-n-and-s.csv`,`synthetic-fix-n-10k.csv`, and `synthetic-fix-ns-480k.csv` respectively in this directory.

Likewise, to generate data for figure 4 for demixing mixed logit on synthetic graph, run:
    ```
    mkdir Mixed-MNL-graph
    python3 generate_Mixed_MNL_graph.py
    python3 Mixed_MNL_model_experiment.py
    ```
    
* Running this will generate another graph in the folder `Mixed-MNL-graph` and run all experiments in the one-mode conditional logit section.
* In `Mixed-MNL_model_experiment.py` there are 2 commands: `fig_4a_4b()` and `fig_4c_4d()`, which can be imported as a module function to a separate file to generate  `synthetic-ml-on-1cl.csv` and `synthetic-ml-on-2cl.csv` respectively in this directory.

The main repository for this paper is located <a href="https://github.com/janovergoor/choose2grow">here</a>.
