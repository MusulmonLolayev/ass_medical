import numpy as np
import sklearn.metrics as metrics
import pickle

np.set_printoptions(precision=4)

from utils.svm_reg_utils \
  import (F_NAMES, load_dataset, train_assessor, one_hot_encoding_system,
        generate_system_features, generate_system, SVMScikit)

def experiment(
        dataset_name: str = 'housing',
        n_systems: int = 50,
        a_acc: int = 0.55,
        b_acc: int = 1,
        fit_from_train: bool = True,
        seed: int = 42,
        eps = np.linspace(0, 1, 10),
        max_error=1):
    
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

    # Generate systems: SMV models
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
      y_pred = model.predict(X_train)
      tr_mae = np.abs(y_train - y_pred)
      x_system[i, F_NAMES['tr_acc']] = np.mean(tr_mae < max_error)
      tr_mae = np.mean(tr_mae)
      x_system[i, F_NAMES['tr_mae']] = tr_mae
      # on the validation
      y_pred = model.predict(X_val)
      val_mae = np.abs(y_val - y_pred)
      x_system[i, F_NAMES['val_acc']] = np.mean(val_mae < max_error)
      val_mae = np.mean(val_mae)
      x_system[i, F_NAMES['val_mae']] = val_mae

      if x_system[i, F_NAMES['val_acc']] < a_acc or \
        x_system[i, F_NAMES['val_acc']] > b_acc:
          y_system[i] = 1

      print(f"MAEs: {tr_mae:.2f}, {val_mae:.2f}")
    
    print(x_system[:, F_NAMES['val_acc']])
        
    # Backup the original systems features, before dropping them
    x_system_backup = x_system.copy()
    y_system_backup = y_system.copy()
    # and save it
    np.save('./file-results/svm_class/x_system_backup.npy', x_system_backup)
    np.save('./file-results/svm_class/y_system_backup.npy', y_system)

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
    print(x_system[:, F_NAMES['val_acc']])

    n_systems = x_system.shape[0]

    # save a new x_system
    np.save('./file-results/svm_class/x_system.npy', x_system)
    np.save('./file-results/svm_class/y_system.npy', x_system)

    # Coverting categorieal variables via one-hot encoding
    one_hot_system = one_hot_encoding_system(x_system)
    print("Number of system features (one-hot):", one_hot_system.shape[1])
    
    ass_model = SVMScikit()
    # Training assessor on the training set
    if fit_from_train:
        train_assessor(X_train, 
                       y_train, 
                       one_hot_system, 
                       systems, 
                       ass_model, 
                       random=True)
    
    # Training assessor on the validation set
    print('='*20, "Training assessor on the validation set", '='*20)
    train_assessor(X_val, 
                   y_val, 
                   one_hot_system, 
                   systems, 
                   ass_model, 
                   random=False, 
                   n_updates=100)
    
    # Producing the second experiment:
    # Select best models for each instance
    s_predictions = np.zeros((X_test.shape[0], n_systems, ))

    print(s_predictions.shape)

    a_predictions = np.zeros((X_test.shape[0], n_systems, ))
    input_x_system = np.zeros((X_test.shape[0], one_hot_system.shape[1]))

    for j in range(n_systems):
        system = systems[j]
        s_predictions[:, j] = system.predict(X_test)
        
        input_x_system[np.arange(X_test.shape[0])] = one_hot_system[j]
        a_predictions[:, j] = ass_model.predict([X_test, input_x_system])[:, 0]

    # prepare data for ploting
    # Get the best and worst accuracies
    best_system_ind = np.argmax(x_system[:, F_NAMES['val_acc']])
    worst_system_ind = np.argmin(x_system[:, F_NAMES['val_acc']])

    # the best and worst systems
    y_best_pred = s_predictions[:, best_system_ind]
    y_best_pred_error = a_predictions[:, best_system_ind]    
    y_worst_pred = s_predictions[:, worst_system_ind]
    y_worst_pred_error = a_predictions[:, worst_system_ind]

    # selected systems by the assessor for each instance
    ass_selected_system_indicies = np.argmin(np.abs(a_predictions), axis=1)
    
    y_sel_pred = s_predictions[np.arange(X_test.shape[0]),
                                        ass_selected_system_indicies]
    y_sel_pred_error = a_predictions[np.arange(X_test.shape[0]),
                                        ass_selected_system_indicies]
    
    y_corr_best_pred = y_best_pred.copy()  
    y_corr_worst_pred = y_worst_pred.copy()  
    y_corr_sel_pred = y_sel_pred.copy()

    best_mae = np.zeros((len(eps), ))
    corr_best_mae = np.zeros((len(eps), ))
    worst_mae = np.zeros((len(eps), ))
    corr_worst_mae = np.zeros((len(eps), ))
    sel_mae = np.zeros((len(eps), ))
    corr_sel_mae = np.zeros((len(eps), ))

    for i, ep in enumerate(eps):
        print(f'Eps: {ep:.2f}, {ep*max_error:.2f}')
        # ep *= max_error
        cond = np.abs(y_best_pred_error) < ep
        y_corr_best_pred[cond] = y_best_pred[cond] + y_best_pred_error[cond]
        cond = np.abs(y_worst_pred_error) < ep
        y_corr_worst_pred[cond] = y_worst_pred[cond] + y_worst_pred_error[cond]
        cond = np.abs(y_sel_pred_error) < ep
        y_corr_sel_pred[cond] = y_sel_pred[cond] + y_sel_pred_error[cond]

        mae = np.mean(np.abs(y_test - y_best_pred))
        corr_mae = np.mean(np.abs(y_test - y_corr_best_pred))
        print(f"Best MAEs: {mae:.2f}/{corr_mae:.2f}")
        best_mae[i] = mae
        corr_best_mae[i] = corr_mae

        mae = np.mean(np.abs(y_test - y_worst_pred))
        corr_mae = np.mean(np.abs(y_test - y_corr_worst_pred))
        print(f"Worst MAEs: {mae:.2f}/{corr_mae:.2f}")
        worst_mae[i] = mae
        corr_worst_mae[i] = corr_mae

        mae = np.mean(np.abs(y_test - y_sel_pred))
        corr_mae = np.mean(np.abs(y_test - y_corr_sel_pred))
        print(f"Selected MAEs: {mae:.2f}/{corr_mae:.2f}")
        sel_mae[i] = mae
        corr_sel_mae[i] = corr_mae

    exp_2 = {
        'best_mae': best_mae,
        'corr_best_mae': corr_best_mae,
        'worst_mae': worst_mae,
        'corr_worst_mae': corr_worst_mae,
        'sel_mae': sel_mae,
        'corr_sel_mae': corr_sel_mae}
    
    print("The second experiment")
    for key, val in exp_2.items():
        print(key, '\n', val)

    # save results as raw pickle file
    with open(f'./file-results/reg/{dataset_name}_exp2.pkl', 'wb') as f:
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
    parser.add_argument('--max-error', default=1, type=float)

    args = parser.parse_args()

    experiment(dataset_name=args.dataset_name,
              n_systems=args.n_systems,
              a_acc=args.a_acc,
              b_acc=args.b_acc,
              seed=args.seed,
              max_error=args.max_error) 