# Predicting Electric Vehicle Purchases

Hi,

This is `September, 2026`.

We have yet another interesting playground competition.

Electric vehicles does solve the problem of **carbon emissions** due to everyday vehicles running on roads.

Tesla and BYD are at the fore-front of the electric vechicle and autonomous vechile development.

What if the product is good and solves a real environmental problem, but the general public doesn't want to migrate towards new technology or product easily.

This competition addresses the very same issue and intends us to create a _prediction model_ around it.

## Evaluation Metric

The evaluation metric is **ROC-AUC**.

*Receiver Operating Characterstic Area Under Curve* tell us how correct is our binary classification model.

Want you understand this concept - learn it on [`numiquo-statistics-explanation`](https://youtu.be/QBVzZBsif20?si=GLkXHHz8i1aNv3gW).

## Leaderboard & Experiments

Here I log the Public Leaderboard (LB) scores of my experiments:

### Baseline

**Kernel:** [shivamgravity/pgs-s6e9-baseline](https://www.kaggle.com/code/shivamgravity/pgs-s6e9-baseline)

| Version | Public LB Score | Notes |
|:---|:---|:---|
| v4 | `0.93738` | Logistic Regression - test_train_split |

