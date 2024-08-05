import numpy as np
import sklearn.metrics as metrics
import pickle

np.set_printoptions(precision=4)

from utils.svm_class_utils \
  import (accuracy_correction, one_hot_encoding_system,
        F_NAMES, load_dataset, train_assessor,
        generate_system_features, generate_system, SVMScikit,
        ass_accuracies)

def experiment(
        dataset_name: str = 'student',
        n_systems: int = 50,
        a_acc: int = 0.55,
        b_acc: int = 0.85,
        fit_from_train: bool = True,
        seed: int = 42,
        eps = np.linspace(0, 0.9, 20)):
    
    # Load dataset
    X_train, X_val, X_test, y_train, y_val, y_test = \
      load_dataset(dataset_name, seed)

    # For just speeding testing
    # indicies = np.random.randint(0, X_train.shape[1], 1000)
    # X_train = X_train[:, indicies]
    # X_val = X_val[:, indicies]
    # X_test = X_test[:, indicies]

    n_train, n_features = X_train.shape
    print(f"Number of objects in train: {n_train}")
    print(f"Number of features: {n_features}")

    # systems' groups: 
    # 0 - the just best groups; 
    # 1 - low error accuracy
    y_system = np.zeros((n_systems), dtype=int)

    # Generate system features
    x_system = generate_system_features(n_systems)

    # Generate systems: simple NN models
    systems = []
    for i in range(n_systems):
        systems.append(generate_system(x_system[i]))
    
    # Train systems
    print('*'*20, 'Training systems', '*'*20)
    for i, model in enumerate(systems):
      print(f"Training model: {i + 1}")
  
      model.fit(X_train, y_train)
      # evaluate models
      # on the train set
      y_pred = model.predict_proba(X_train)
      y_pred = np.argmax(y_pred, axis=1)
      tr_acc = np.mean(y_train == y_pred)
      x_system[i, F_NAMES['tr_acc']] = tr_acc
      # on the validation
      y_pred = model.predict_proba(X_val)
      y_pred = np.argmax(y_pred, axis=1)
      val_acc = np.mean(y_val == y_pred)
      x_system[i, F_NAMES['val_acc']] = val_acc

      if x_system[i, F_NAMES['val_acc']] < a_acc or \
        x_system[i, F_NAMES['val_acc']] > b_acc:
          y_system[i] = 1

      print(f"Accuracies: {tr_acc:2f}, {val_acc:2f}")
        
    print("Remove systems with low interval error accuracy")
    # Remove systems with low interval error accuracy
    # for avoiding waiting much time to compute
    # Filtering systems
    new_models = []
    for i, model in enumerate(systems):
        if y_system[i] == 0:
          new_models.append(model)
    systems = new_models

    x_system = x_system[y_system == 0]
    y_system = y_system[y_system == 0]
    # New system shape
    print(f"Initially, num. systems: {n_systems}, Now, {x_system.shape[0]}")
    n_systems = x_system.shape[0]

    # Coverting categorieal variables via one-hot encoding
    one_hot_system = one_hot_encoding_system(x_system)
    # print("Number of system features (one-hot):", one_hot_system.shape[1])
    # one_hot_system = x_system
    
    ass_model = SVMScikit()
    # Training assessor on the training set
    n_labels = y_train.max() + 1

    if fit_from_train:
        train_assessor(X_train, 
                       y_train, 
                       one_hot_system, 
                       systems, 
                       ass_model,
                       n_labels,
                       random=True)
    
    # Training assessor on the validation set
    print('='*20, "Training assessor on the validation set", '='*20)
    train_assessor(X_val, 
                   y_val, 
                   one_hot_system, 
                   systems, 
                   ass_model,
                   n_labels,
                   random=False, 
                   n_updates=2)
    
    # Producing the second experiment:
    # Select best models for each instance
    s_predictions = np.zeros((X_test.shape[0], n_systems, n_labels))

    a_predictions = np.zeros((X_test.shape[0], n_systems, n_labels))
    input_x_system = np.zeros((X_test.shape[0], one_hot_system.shape[1]))

    for j in range(n_systems):
        system = systems[j]
        s_predictions[:, j, :] = system.predict_proba(X_test)
        
        input_x_system[np.arange(X_test.shape[0])] = one_hot_system[j]
        a_predictions[:, j, :] = ass_model.predict([X_test, input_x_system])
    
    # prepare data for ploting
    # Get the best and worst accuracies
    best_system_ind = np.argmax(x_system[:, F_NAMES['val_acc']])
    worst_system_ind = np.argmin(x_system[:, F_NAMES['val_acc']])

    # the best and worst systems
    y_best_pred = s_predictions[:, best_system_ind, :]
    y_best_pred_error = a_predictions[:, best_system_ind, :]    
    y_worst_pred = s_predictions[:, worst_system_ind, :]
    y_worst_pred_error = a_predictions[:, worst_system_ind, :]

    # selected systems by the assessor for each instance
    ass_selected_system_indicies = np.argmin(
        np.sum(np.abs(a_predictions), axis=-1),
        axis=1)
    
    y_sel_pred = s_predictions[np.arange(X_test.shape[0]),
                                        ass_selected_system_indicies]
    y_sel_pred_error = a_predictions[np.arange(X_test.shape[0]),
                                        ass_selected_system_indicies]
    best_accs = np.zeros((len(eps), ))
    corr_best_accs = np.zeros((len(eps), ))
    worst_accs = np.zeros((len(eps), ))
    corr_worst_accs = np.zeros((len(eps), ))
    sel_accs = np.zeros((len(eps), ))
    corr_sel_accs = np.zeros((len(eps), ))

    ass_best_accs = np.zeros((len(eps), ))
    ass_worst_accs = np.zeros((len(eps), ))

    for i, ep in enumerate(eps):
        y_best_pred_label, y_corr_best_pred_label = \
          accuracy_correction(y_best_pred, y_best_pred_error, ep)
        y_worst_pred_label, y_corr_worst_pred_label = \
          accuracy_correction(y_worst_pred, y_worst_pred_error, ep)
        y_sel_pred_label, y_sel_worst_pred_label = \
          accuracy_correction(y_sel_pred, y_sel_pred_error, ep)
        
        best_accs[i] = metrics.accuracy_score(y_test, y_best_pred_label)
        corr_best_accs[i] = metrics.accuracy_score(y_test, y_corr_best_pred_label)
        worst_accs[i] = metrics.accuracy_score(y_test, y_worst_pred_label)
        corr_worst_accs[i] = metrics.accuracy_score(y_test, y_corr_worst_pred_label)
        sel_accs[i] = metrics.accuracy_score(y_test, y_sel_pred_label)
        corr_sel_accs[i] = metrics.accuracy_score(y_test, y_sel_worst_pred_label)

        ass_best_accs[i] = ass_accuracies(y_best_pred, 
                                          y_best_pred_error, 
                                          y_test,
                                          ep)
        ass_worst_accs[i] = ass_accuracies(y_worst_pred, 
                                          y_worst_pred_error, 
                                          y_test,
                                          ep)
        

    exp_2 = {
        'best_accs': best_accs,
        'corr_best_accs': corr_best_accs,
        'worst_accs': worst_accs,
        'corr_worst_accs': corr_worst_accs,
        'sel_accs': sel_accs,
        'corr_sel_accs': corr_sel_accs,
        'ass_best_accs': ass_best_accs,
        'ass_worst_accs': ass_worst_accs}
    
    print("The second experiment")
    for key, val in exp_2.items():
        print(key, '\n', val)

    # save results as raw pickle file
    with open(f'./file-results/softmax/{dataset_name}_exp2.pkl', 'wb') as f:
        pickle.dump(exp_2, f)

if __name__ == '__main__':
    

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', 
                        default='students', 
                        type=str)
    parser.add_argument('--n_systems', default=50, type=int)
    parser.add_argument('--a_acc', default=0.5, type=float)
    parser.add_argument('--b_acc', default=0.85, type=float)
    parser.add_argument('--seed', default=42, type=int)

    args = parser.parse_args()

    experiment(dataset_name=args.dataset_name,
              n_systems=args.n_systems,
              a_acc=args.a_acc,
              b_acc=args.b_acc,
              seed=args.seed)    