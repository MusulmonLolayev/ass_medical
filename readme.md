# Assessor meta-models for safety and accuracy in medical AI systems

## Abstract

Establishing robust guardrails and oversight mechanisms for AI models, especially in medicine, is now viewed as an essential obligation by the research community, prompted by the multitude of challenges and failures that have materialized within this sector throughout the last decade. An assessor meta-model can serve as a safeguard by scrutinizing AI models in advance and quantifying their likelihood of effective performance on a given target task or dataset. This probabilistic analysis provides an additional layer of oversight prior to employing any machine learning model. The present work leverages the assessor meta-model approach to establish guardrails for machine learning models tailored for medical use cases. This safeguarding process is implemented by evaluating the models' predicted success rates on several healthcare-related datasets, facilitating a robust screening mechanism. Modifications introduced to the initial proposal achieved a dual benefit - elevating the precision of the models alongside fortifying their safeguards.

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