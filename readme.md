# Assessor meta-models for safety and accuracy in medical AI systems
## 1. Introduction
Recently assessor models have been proposed to explain Artificial System behavior on a given task. In this paper, we propose a methodology which we call "Assessor feedback mechanism" to leverage the power of assessors to correct AI outputs. Assume we have several AI robots to deliver goods around a city, we do not know which robots can do a certain job successfully until employ one of them and get their results. In this case, we would propose to use another AI model called assessor model to assess them on a particular job. Instead learning ML model a success/failure probability introduced by authors of this assessors on a given task:

$$\hat{R}(r|x_j, s_i)\approx Pr(R(x_j, s_i)=r)$$

where $x_j$ - instance features is being to predicted by an ML model with emergent behavior and other related factors to the prediction process characterized by $s_i$ (simple the emergent behavior of the model). 

We would train assessor models on AI models' errors on their (AI model) prediction to correct their errors. For example, for regression task, let's define some notations: a model set $s_i \in S$, and their errors $e_j^i=y_j-\hat{y}_j^i$ ($x_i$ object predicted by model $s_j$ with prediction result $\hat{y}_j^i$, and the prediction error is $e_j^i$) on a set of instances $x_j \in X$ with target values $y_j$. Now, our assessor model is going to learn errors $e_j \in E$. The following assessor model then predict the error on new input instance $x_{n+1}$ and system $s_{m+1}$ (a new system can be unseen by the assessor model yet).

$$\hat{e}=\hat{E}(x_j, s_i)$$

To choose the best model on test or production mode, we first predict a error produced by an AI model when predicting a given instance (we predict the error using the assessor model), and we then find a model $s_i$ which produces the lowest error among all exists AI models on a new input $x_j$, and then we predict object $x_j$ by system $s_i$, and finally correct a error $e_j^i$ produced by $s_i$ on instance $x_j$.

$$\hat{y}_j^i=Model_i(x_i)+e_j^i$$

By using the same framework, we achieve doing two staff:

- assess model error on a particular instance if low (or desirable), we can employ a system with the lowest error;
- we can correct the ML model prediction by adding the error produced by the assessor model.
## 2. Datasets

All datasets stored in folder './datasets'.

## 3. Run the experiments

### 3.1 Setup the environment

To install requirements, you can use a conda yaml file or a standard pip requirements.txt file located in the main folder.

### 3.2 Run experiment scripts

#### 3.2.1 SVM classification

```
python3 ./exp_svm_class.py --dataset_name breast_cancer --n_systems 30 --a_acc 0.55 --b_acc 0.85 --seed 42
```

#### 3.2.1 SVM regression


```
python3 ./exp_svm_reg.py --dataset_name parkinsons_total --n_systems 30 --a_acc 0.55 --b_acc 0.85  --max-error 2 --seed 42
```

Note all hyper-parameters should be adjusted to get the same results with in the paper.