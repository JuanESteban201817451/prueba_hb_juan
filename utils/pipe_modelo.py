import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os
from tqdm import tqdm
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.neural_network import MLPRegressor
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
import optuna
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def objective_xgboost(trial, X, y): 
    param = {
        "verbosity": 0,
        "booster": trial.suggest_categorical("booster", ["gbtree", "dart"]),
        "lambda": trial.suggest_float("lambda", 1e-4, 1.0, log=True),
        "alpha": trial.suggest_float("alpha", 1e-4, 1.0, log=True),
        "eta": trial.suggest_float("eta", 0.01, 0.2, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 6),
        "subsample": trial.suggest_float("subsample", 0.5, 0.9),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 0.9),
        "min_child_weight": trial.suggest_float("min_child_weight", 1, 10),
        "tree_method": "hist",
    }
    dtrain = xgb.DMatrix(X, label=y)
    model = xgb.train(param, dtrain, num_boost_round=100, verbose_eval=False)
    preds = model.predict(dtrain)
    return np.sqrt(mean_squared_error(y, preds))

def objective_random_forest(trial, X, y):
    param = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 400),
        'max_depth': trial.suggest_int('max_depth', 4, 60),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
    }
    model = RandomForestRegressor(**param)
    return -cross_val_score(model, X, y, scoring='neg_root_mean_squared_error', cv=3).mean()

def objective_lightgbm(trial, X, y):
    param = {
        'num_leaves': trial.suggest_int('num_leaves', 31, 128),
        'learning_rate': trial.suggest_float('learning_rate',  0.01, 0.2),
        'n_estimators': trial.suggest_int('n_estimators', 100, 600),
    }
    model = lgb.LGBMRegressor(**param)
    return -cross_val_score(model, X, y, scoring='neg_root_mean_squared_error', cv=3).mean()

def objective_catboost(trial, X, y):
    param = {
        'iterations': trial.suggest_int('iterations', 100, 600),
        'depth': trial.suggest_int('depth', 4, 8),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
        'verbose': 0
    }
    model = cb.CatBoostRegressor(**param)
    return -cross_val_score(model, X, y, scoring='neg_root_mean_squared_error', cv=3).mean()

def objective_ridge(trial, X, y):
    alpha = trial.suggest_float('alpha', 0.01, 10.0)
    model = Ridge(alpha=alpha)
    return -cross_val_score(model, X, y, scoring='neg_root_mean_squared_error', cv=3).mean()

def objective_mlp(trial, X, y):
    param = {
        'hidden_layer_sizes': tuple([trial.suggest_int('units_l%d' % i, 10, 50) for i in range(2)]),
        'activation': trial.suggest_categorical('activation', ['relu', 'tanh']),
        'alpha': trial.suggest_float('alpha', 1e-5, 1e-1, log=True),
        'learning_rate_init': trial.suggest_float('learning_rate_init', 1e-4, 1e-1, log=True),
        'max_iter': 300
    }
    model = MLPRegressor(**param)
    return -cross_val_score(model, X, y, scoring='neg_root_mean_squared_error', cv=3).mean()

def ejecutar_pipeline(df, target_col='precio'):
    os.makedirs("data/procesado", exist_ok=True)

    df = df[df[target_col] <= 1_100_000_000].copy()
    df = df[(df['area'] >= 20) & (df['area'] <= 1000 )].copy()
    print(len(df))
    cat_cols = ['tiponegocio', 'CODIGO_UPL', 'banos', 'garajes', 'num_ascensores','anno_creacion', 'anos_antiguedad']
    num_cols = ['area', 'habitaciones','latitud','longitud']

    df = df.dropna(subset=cat_cols + num_cols + [target_col])

    for tipo in ['venta', 'arriendo']:
        print(f"Entrenando modelo para: {tipo.upper()}")
        df_sub = df[df['tiponegocio'] == tipo].copy()
        print(len(df_sub))

        X = df_sub[cat_cols + num_cols]
        y = np.log1p(df_sub[target_col])
        y_real = df_sub[target_col]

        X_train, X_test, y_train, y_test, y_train_real, y_test_real = train_test_split(
            X, y, y_real, test_size=0.2, random_state=42, stratify=df_sub['CODIGO_UPL'])

        df_train = X_train.copy()
        df_train[target_col] = y_train_real
        df_test = X_test.copy()
        df_test[target_col] = y_test_real

        df_train.to_csv(f"data/procesado/train_{tipo}.csv", index=False)
        df_test.to_csv(f"data/procesado/test_{tipo}.csv", index=False)

        resultados = {}
        modelos = {
            'xgboost': (objective_xgboost, False),
            'lightgbm': (objective_lightgbm, False),
            'catboost': (objective_catboost, False),
            'ridge': (objective_ridge, True),
            'mlp': (objective_mlp, True),
            'random_forest': (objective_random_forest, False),
        }

        for nombre, (objective, necesita_escalado) in tqdm(modelos.items(), desc=f"Modelos {tipo}"):
            transformers = [('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)]
            if necesita_escalado:
                transformers.append(('num', StandardScaler(), num_cols))
            else:
                transformers.append(('num', 'passthrough', num_cols))

            preprocessor = ColumnTransformer(transformers)

            X_train_proc = preprocessor.fit_transform(X_train)
            X_test_proc = preprocessor.transform(X_test)

            study = optuna.create_study(direction='minimize')
            study.optimize(lambda trial: objective(trial, X_train_proc, y_train), n_trials=40)

            best_params = study.best_params

            if nombre == 'xgboost':
                model = xgb.train(best_params, xgb.DMatrix(X_train_proc, label=y_train))
                preds_log = model.predict(xgb.DMatrix(X_test_proc))
            elif nombre == 'catboost':
                model = cb.CatBoostRegressor(**best_params)
                model.fit(X_train_proc, y_train)
                preds_log = model.predict(X_test_proc)
            elif nombre == 'lightgbm':
                model = lgb.LGBMRegressor(**best_params)
                model.fit(X_train_proc, y_train)
                preds_log = model.predict(X_test_proc)
            elif nombre == 'random_forest':
                model = RandomForestRegressor(**best_params)
                model.fit(X_train_proc, y_train)
                preds_log = model.predict(X_test_proc)
            elif nombre == 'ridge':
                model = Ridge(**best_params)
                model.fit(X_train_proc, y_train)
                preds_log = model.predict(X_test_proc)
            elif nombre == 'mlp':
                model = MLPRegressor(**best_params)
                model.fit(X_train_proc, y_train)
                preds_log = model.predict(X_test_proc)

            if np.any(np.isnan(preds_log)) or np.any(np.isinf(preds_log)):
                print(f"Advertencia: predicciones inválidas en {nombre} - {tipo}. Se omite este modelo.")
                continue

            preds = np.expm1(preds_log)

            print(f"{nombre} - {tipo}: Predicciones expuestas - max: {preds.max():,.2f}, min: {preds.min():,.2f}")

            rmse = np.sqrt(mean_squared_error(y_test_real, preds))
            mape = mean_absolute_percentage_error(y_test_real, preds)
            resultados[nombre] = {
                'model': model, 'rmse': rmse, 'mape': mape,
                'tipo': tipo, 'modelo': nombre
            }

            df_individual = pd.DataFrame({
                'modelo': [nombre],
                'tipo': [tipo],
                'RMSE': [rmse],
                'MAPE': [mape]
            })
            df_individual.to_csv(f"data/procesado/resumen_{nombre}_{tipo}.csv", index=False)

            joblib.dump(model, f"data/procesado/modelo_final_{tipo}_{nombre}.pkl")
            print(f"Modelo guardado: data/procesado/modelo_final_{tipo}_{nombre}.pkl\n")

        df_resultados = pd.DataFrame({k: {'RMSE': v['rmse'], 'MAPE': v['mape']} for k, v in resultados.items()}).T.sort_values('RMSE')
        df_resultados.to_csv(f"data/procesado/rmse_mape_modelos_{tipo}.csv")