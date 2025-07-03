import xgboost
import numpy as np
import pandas as pd 
import sys 
import pickle

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
import hyperopt

import data
from model import build_and_optimize_model

import torch

pd.options.mode.chained_assignment = None



def obj_ratio_fn(y_calc,y_exp):
    '''
    Custom loss function for the ratio technique
    Inputs:
        y_calc - calculation of experimental property
        y_exp - experimental property
    Outputs:
        ratio_loss - loss value from ratio technique
    '''

    def ratio_loss(y_true,y_pred) -> np.ndarray:

        gradient = 2.0*y_calc*(y_pred*y_calc - y_exp)
        hessian = 2.0*(y_calc**2) + 0.0*y_calc*(y_pred*y_calc - y_exp)

        return gradient, hessian
    
    return ratio_loss


def run_active_learning(total_df,theory_method,save_file,objective='MaxVar'):
    '''
    Active learning algorithm using GPR and an acquisition function.
    Inputs: 
        total_df - exploration space (data points) in a Pandas dataframe
        theory_method - technique for theory-ML model
        save_file - filename to save results to
        objective - use 'MaxVar' or 'MaxDiff' acquisition functions
    Outputs:
        None. Files are saved to specificed save_file location. 
    '''

    all_rmse_scores = []
    all_mae_scores = []
    all_r2_scores = []

    for j in range(1,11):
        rmse_scores = []
        mae_scores = []
        r2_scores = []

        split_df = total_df.copy()
        split_y = split_df.pop('Experimental_deltaG')

        X_train, X_test, y_train, y_test = train_test_split(split_df,split_y,train_size=20,random_state=42*j)

        X_train = X_train.reset_index(drop=True) #need to reset so that test set can be selected by list of index values
        X_test = X_test.reset_index(drop=True) #need to reset so that test set can be selected by list of index values

        if theory_method != 'informed':
            y_train_calc = X_train.pop('Calculated_deltaG')
            y_test_calc = X_test.pop('Calculated_deltaG')

        next_sample = None
        for i in range(0,201):

            if next_sample is not None:
                X_train = pd.concat([X_train,X_test.iloc[next_sample]],ignore_index=True).reset_index(drop=True)
                y_train = pd.concat([y_train,y_test.iloc[next_sample]],ignore_index=True).reset_index(drop=True)
                
                X_test = X_test.drop(X_test.index[next_sample]).reset_index(drop=True)
                y_test = y_test.drop(y_test.index[next_sample]).reset_index(drop=True)

                if theory_method != 'informed':  
                    y_train_calc = pd.concat([y_train_calc,y_test_calc.iloc[next_sample]],ignore_index=True).reset_index(drop=True)              
                    y_test_calc = y_test_calc.drop(y_test_calc.index[next_sample]).reset_index(drop=True)

            y_train_exp = y_train.copy()
            y_test_exp = y_test.copy().values

            if theory_method == 'diff':
                y_train_vals = y_train_exp - y_train_calc
            else:
                y_train_vals = y_train_exp
            
            scaler_x = StandardScaler()
            X_scaled = scaler_x.fit_transform(X_train.copy())
            #y_scaled = scaler_y.fit_transform(y_train_vals.values.reshape(-1,1))
            y_scaled = y_train_vals.values
            
            X_train_tensor = torch.tensor(X_scaled, dtype=torch.float32)
            y_train_tensor = torch.tensor(y_scaled, dtype=torch.float32).reshape(-1, 1).squeeze()

            #xgb = xgboost.XGBRegressor(objective=obj_diff_fn(y_train_calc,y_train_exp))
            model, likelihood = build_and_optimize_model(X_train_tensor,y_train_tensor)

            X_test_scaled = scaler_x.transform(X_test.copy())
            X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)

            if theory_method == 'diff':
                predictions = likelihood(model(X_test_tensor)).mean.detach().numpy() + y_test_calc.values
            else:
                # predictions = scaler_y.inverse_transform(likelihood(model(X_test_tensor)).mean.numpy().reshape(-1,1)).reshape(-1,)
                predictions = likelihood(model(X_test_tensor)).mean.detach().numpy()

            # variances = scaler_y.inverse_transform(likelihood(model(X_test_tensor)).variance.numpy().reshape(-1,1)).reshape(-1,)
            variances = likelihood(model(X_test_tensor)).variance.detach().numpy().reshape(-1,)

            rmse_scores.append(np.sqrt(np.square(predictions-y_test_exp).mean()).tolist())
            mae_scores.append(mean_absolute_error(y_test_exp,predictions))
            r2_scores.append(r2_score(y_test_exp,predictions))
            
            if objective=='MaxVar':
                next_sample = np.argsort(variances)[-5:]
                print(i,rmse_scores[-1],np.mean(variances),np.max(variances),variances[next_sample],next_sample)
            if objective=='MaxDiff':
                if theory_method == 'informed':
                    y_test_calc = X_test['Calculated_deltaG'].copy()
                next_sample = np.argsort(np.absolute(predictions-y_test_calc.values))[-5:]
                if j==1 and i==0:
                    print("Iteration","RMSE","Mean Difference","Max Difference","Mean Variance","Max Variance","Sampled Difference 1", "Sampled Difference 2", "Sampled Difference 3", "Sampled Difference 4", "Sampled Difference 5", "Next Sample Indices")
                print(i,rmse_scores[-1],np.mean(np.absolute(predictions-y_test_calc.values)),np.max(np.absolute(predictions-y_test_calc.values)),np.mean(variances),np.max(variances),np.absolute(predictions-y_test_calc.values)[next_sample],next_sample)
            if objective=='Random':
                next_sample = X_test.sample(n=5,random_state=42*j).index.values.tolist()
                print(i,rmse_scores[-1],np.mean(variances),np.max(variances),variances[next_sample],next_sample)

        all_rmse_scores.append(rmse_scores)
        all_mae_scores.append(mae_scores)
        all_r2_scores.append(r2_scores)


    np.savetxt('%s_GPR_calc_rmse_%s.csv'%(theory_method,save_file), np.array(all_rmse_scores), delimiter=',',fmt='%.8f')
    np.savetxt('%s_GPR_calc_mae_%s.csv'%(theory_method,save_file), np.array(all_mae_scores), delimiter=',',fmt='%.8f')
    np.savetxt('%s_GPR_calc_r2_%s.csv'%(theory_method,save_file), np.array(all_r2_scores), delimiter=',',fmt='%.8f')

    return

def loss_vs_data(total_df,theory_method,model_file,save_file):
    '''
    Calculate loss vs. number of training data (randomly chosen). 
    Inputs:
        total_df - dataset represented as a Pandas dataframe
        theory_method - technique for theory-ML model
        model_file - location of file for optimized model hyperparameters
        save_file - filename to save results to
    Outputs:
        None. Files are saved to specificed save_file location. 
    '''

    all_rmse_scores = []
    all_mae_scores = []
    all_r2_scores = []

    num_data_list = [20, 50]
    percent_data_list = [i/100 for i in range(5,100,5)]
    total_data_list = np.concatenate([num_data_list,percent_data_list])

    for j in range(1,11):
        rmse_scores = []
        mae_scores = []
        r2_scores = []

        split_df = total_df.copy()
        split_y = split_df.pop('Experimental_deltaG')

        for i in total_data_list:

            if i > 1:
                train_size = int(i)
            else:
                train_size = i
            
            X_train, X_test, y_train, y_test = train_test_split(split_df,split_y,train_size=train_size,stratify=df_copy['Stratify_by_rotors'],random_state=42*j)
            
            y_train_exp = y_train.values
            y_test_exp = y_test.values

            if theory_method != 'informed':
                y_train_calc = X_train.pop('Calculated_deltaG').values
                y_test_calc = X_test.pop('Calculated_deltaG').values

            if theory_method == 'diff':
                y_train_vals = y_train_exp - y_train_calc
            else:
                y_train_vals = y_train_exp
            
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_train)

            with open(model_file,'rb') as file:
                model_params = pickle.load(file)

            if theory_method == 'ratio':
                xgb = xgboost.XGBRegressor(objective=obj_ratio_fn(y_train_calc,y_train_exp),**model_params)
            else:
                xgb = xgboost.XGBRegressor(**model_params)

            xgb.fit(X_scaled,y_train_vals)

            X_test_scaled = scaler.transform(X_test)
            
            predictions = xgb.predict(X_test_scaled)

            if theory_method == 'diff':
                predictions += y_test_calc
            elif theory_method == 'ratio':
                predictions *= y_test_calc
            else:
                pass

            rmse_scores.append(np.sqrt(np.square(predictions-y_test_exp).mean()))
            mae_scores.append(mean_absolute_error(y_test_exp,predictions))
            r2_scores.append(r2_score(y_test_exp,predictions))
            
        all_rmse_scores.append(rmse_scores)
        all_mae_scores.append(mae_scores)
        all_r2_scores.append(r2_scores)


    np.savetxt('./Loss_vs_Data_Results/%s_calc_rmse_%s.csv'%(theory_method,save_file), np.array(all_rmse_scores), delimiter=',',fmt='%.8f')
    np.savetxt('./Loss_vs_Data_Results/%s_calc_mae_%s.csv'%(theory_method,save_file), np.array(all_mae_scores), delimiter=',',fmt='%.8f')
    np.savetxt('./Loss_vs_Data_Results/%s_calc_r2_%s.csv'%(theory_method,save_file), np.array(all_r2_scores), delimiter=',',fmt='%.8f')

    return 

def main(params):
    '''
    Main optimization function for Hyperopt.
    Inputs:
        params - Hyperopt parameters selected for current iteration
    Outputs: 
        Dictionary of Hyperopt results
    '''

    theory_method = params.pop('theory_method')

    y_train_calc = split_y_calc.loc[X_train.index.values.tolist()].copy()
    X_train_copy = df_copy.loc[X_train.index.values.tolist()].copy()

    skf = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)
    cv_scores = []

    for i, (train_index, test_index) in enumerate(skf.split(X_train, X_train_copy['Stratify_by_rotors'])):

        X_train_fold = X_train.iloc[train_index]
        y_train_fold = y_train.iloc[train_index]

        X_val_fold = X_train.iloc[test_index]
        y_val_fold = y_train.iloc[test_index]

        y_calc_train_fold = y_train_calc.iloc[train_index]
        y_calc_val_fold = y_train_calc.iloc[test_index]

        if theory_method == 'diff':
            y_train_fold = y_train_fold - y_calc_train_fold
        elif theory_method == 'informed':
            X_train_fold.loc[:,'Calculated_deltaG'] = y_calc_train_fold.values.tolist()
            X_val_fold.loc[:,'Calculated_deltaG'] = y_calc_val_fold.values.tolist()
        else:
            pass
        
        if theory_method == 'ratio':
            xgbmodel = xgboost.XGBRegressor(objective=obj_ratio_fn(y_calc_train_fold.values,y_train_fold.values),n_jobs=-1,tree_method='hist',**params)
        else:
            xgbmodel = xgboost.XGBRegressor(n_jobs=-1,tree_method='hist',**params)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_train_fold)

        if theory_method=='surrogate':
            xgbmodel.fit(X_scaled,y_calc_train_fold)
        else:
            xgbmodel.fit(X_scaled,y_train_fold)
        
        X_test_scaled = scaler.transform(X_val_fold)
        predictions = xgbmodel.predict(X_test_scaled)
        
        if theory_method == 'diff':
            predictions += y_calc_val_fold
        elif theory_method == 'ratio':
            predictions *= y_calc_val_fold
        else:
            pass

        cv_scores.append(mean_squared_error(y_val_fold,predictions))

    sys.stdout.flush() 
    return {'loss': np.mean(cv_scores), 'status': STATUS_OK, 'model': xgbmodel}


def run(run_hyperopt,active_learning,save_file,theory_method=None,theory_column='DeltaGsolv uESE (kcal/mol) - 1',objective='MaxVar'):
    '''
    Primary function to calculate results.
    Inputs:
        run_hyperopt - Boolean flag to run Hyperopt optimization
        active_learning - Boolean flag to run active learning algorithm
        save_file - filename including path to save results
        theory_method - technique for theory-ML model
        theory_column - name of Pandas dataframe column that contains theoretical calculation
        objective - (for active learning only) name of acquitision function ('MaxVar' or 'MaxDiff')
    '''

    if run_hyperopt:

        global X_train, y_train, split_y_calc, df_copy

        total_df, df_copy = data.load_minnesota_data(theory_column)

        total_df = total_df.drop(columns=['Solvent SMILES_solvent','Solute SMILES_solute'])

        split_df = total_df.copy()
        split_y = split_df.pop('Experimental_deltaG')
        split_y_calc = split_df.pop('Calculated_deltaG')

        print("Number of solute/solvent pairs:",len(split_df))
        print("Standard deviation of free energy values:",np.std(split_y_calc.values))


        X_train, X_test, y_train, y_test = train_test_split(split_df,split_y,train_size=0.8,test_size=0.2,stratify=df_copy['Stratify_by_rotors'],random_state=42)
        
        xgboost_space = {
            'n_estimators': hp.choice('n_estimators', np.arange(start=100,stop=501,step=10)),
            'eta': hp.choice('eta',np.arange(start=1/100, stop=1/2 + 1/100, step=1/100)),
            'gamma': hp.choice('gamma',np.arange(start=0, stop=10+1/2, step=1/2)),
            'max_depth':  hp.choice('max_depth',np.arange(start=1, stop=21, step=1)),
            'min_child_weight': hp.choice('min_child_weight',np.arange(start=1, stop=6, step=1)),
            'subsample': hp.choice('subsample',np.arange(start=1/2, stop=1 + 5/100, step=1/10)),
            'colsample_bytree': hp.choice('colsample_bytree',np.arange(start=1/2,stop=1 + 5/100, step=1/10)),
            'reg_lambda': hp.choice('reg_lambda',np.arange(start=0,stop=10 + 1/2,step=1/2)),
            'random_state': 42,
            'theory_method': theory_method
        }

        print('Starting hyperopt...')
        trials = Trials()
        best_params = fmin(main, xgboost_space, algo=tpe.suggest, max_evals=100, trials=trials, rstate=np.random.default_rng(42))
        print('Done!')

        print('Best parameters = ', hyperopt.space_eval(xgboost_space, best_params))

        # Evaluate model
        loss = trials.best_trial['result']['loss']
        print ('Best loss :', loss)

        best_model = trials.best_trial['result']['model']
        best_params = best_model.get_params()
        best_params.pop('objective')

        if theory_method is None:
            theory_method = 'MLonly'

        filename = './xgboost_hyperopt_5foldCV_%s_%s.pkl'%(theory_method,save_file)

        with open(filename,'wb') as file:
            pickle.dump(best_params,file)

    if active_learning:

        if theory_method is None:
            theory_method = 'MLonly'

        combined_df = data.load_combined_data()
        combined_df = combined_df.drop(columns=['Solvent SMILES_solvent','Solute SMILES_solute'])

        run_active_learning(combined_df,theory_method,save_file,objective)

    if (not active_learning) and (not run_hyperopt):
        print("Running Loss vs. Number of Data Calculation...")

        total_df, df_copy = data.load_minnesota_data()
        total_df = total_df.drop(columns=['Solvent SMILES_solvent','Solute SMILES_solute'])

        if theory_method is None:
            theory_method = 'MLonly'
            model_file = './xgboost_hyperopt_5foldCV_%s.pkl'%theory_method
        else:
            model_file = './xgboost_hyperopt_5foldCV_%s_uESE1.pkl'%theory_method
        
        loss_vs_data(total_df,theory_method,model_file,save_file)
        print("Done")


if __name__ == '__main__':
    save_file = 'MaxDiff'
    run_hyperopt = False
    active_learning = True
    run(run_hyperopt,active_learning,save_file=save_file,theory_method='informed',theory_column='DeltaGsolv uESE (kcal/mol) - 1', objective='MaxDiff')