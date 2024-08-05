# For regression experiment
# For SVM classifier
python3 ./exp_svm_class.py --dataset_name breast_cancer --n_systems 30 --a_acc 0.55 --b_acc 0.85 --seed 42

# For SVM Regression
python3 ./exp_svm_reg.py --dataset_name parkinsons_total --n_systems 30 --a_acc 0.55 --b_acc 0.85  --max-error 2 --seed 42