from datetime import datetime
import optuna


def xgb_objective_func(
    trial,
    features: pd.DataFrame,
    target: pd.Series,
) -> float:
    """
    cb_objective function that returns mean test R2 - STD test R2 from K-Fold CV.
    """

    # fill in more via https://catboost.ai/en/docs/references/training-parameters/common#bootstrap_type
    param = {
        'reg_alpha': 0.000517878113716743,
        'reg_lambda': 0.00030121415155097723,
        'gamma': 5.301218558776368e-08,
        'subsample': 0.41010429946197946,
        'n_jobs': -1,
        'verbosity': 0,
        'eval_metric': 'logloss',
        'objective': 'binary:logistic',
        'max_depth': trial.suggest_int('max_depth', 2, 6),
        'learning_rate': trial.suggest_float('learning_rate', 0.02, 0.1),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 1.0),
        'min_child_weight': trial.suggest_int("depth", 3, 10),
        'num_boost_round': trial.suggest_int('num_boost_round', 200, 1000),
        'early_stopping_rounds': trial.suggest_int('early_stopping_rounds', 5, 100),
    }

    print(param)
    return xgb_log_loss_kfold(
        x_data=features,
        y_data=target,
        classifier_params=param,
    )


# run the optimization
xgb.config_context(verbosity=0)
RUN_XGB_OPTIMIZATION = False

if RUN_XGB_OPTIMIZATION:
    n_trials = 25
    timeout = None

    # create the optuna study
    print(
        f'Starting XGBoost optimization w/ n_trials={n_trials}, and timeout={timeout}'
    )

    # record time
    start_time = datetime.now()

    # workaround to pass argument into the cb_objective https://www.kaggle.com/general/261870
    def xgb_objective(trial): return xgb_objective_func(
        trial,
        train_df.drop(columns=['Class']),
        train_df['Class'],

    )

    # run the study
    study = optuna.create_study(direction="minimize")
    study.optimize(
        xgb_objective,
        n_trials=n_trials,
        timeout=None,
    )

    # record end time and key results
    end_time = datetime.now()
    print(
        f'XGBoost Optimization complete! Took {end_time - start_time}.\n'
    )

    # log key stats
    best_trial = study.best_trial
    print(f'Number of finished trials: {len(study.trials)}')

    print(f'Best trial = {best_trial}')
    print(f'Best trial value = {best_trial.value}\n')

    print(f'Best hyperparameters:')

    for key, value in best_trial.params.items():
        print(f'{key} = {value}')
    print('-------------------------------------------\n\n')
else:
    print('NOT running optuna optimization bc RUN_CB_OPTIMIZATION=False')
