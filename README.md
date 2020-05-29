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

The main repository for this paper is located <a href="https://github.com/janovergoor/choose2grow">here</a>.
