# RSA_recommender-system-algorithms
learning notes about recommender system algorithms




## Code Structure

The structure of our project is presented in a tree form as follows:

```
Recommender System  # the root of project
│   README.md
│   __init__.py
│   .gitignore
|
└───configx  # configurate the global parameters and hyper parameters
│   │   configx.py   
|   │   
└───data  # store the rating and social data
│   │   ft_ratings.txt
|   │   ft_trust.txt
|   |
│   └───cv  # cross validation data
│       │   ft-0.txt
│       │   ft-1.txt
│       │   ft-2.txt
│       │   ft-3.txt
│       │   ft-4.txt
|       |
└───metrics  # the metrics to measure the prediction accuracy for rating prediction task
│   │   metric.py
|   |
└───model  # the set of methods of tranditional and social recommendation
│   │   bias_svd.py
│   │   funk_svd.py
│   │   pmf.py
│   │   integ_svd.py
|   |   item_cf.py
|   |   item_cf_big.py
|   |   mf.py
|   |   social_mf.py
|   |   social_rec.py
|   |   social_reg.py
|   |   social_rste.py
|   |   svd++.py
|   |   trust_svd.py
|   |   trust_walker.py
|   |   user_cf.py
|   |
└───reader  # data generator for rating and social data
│   │   rating.py
│   │   trust.py
|   |
└───utility  # other commonly used tools
    │   cross_validation.py
    │   data_prepro.py
    │   data_statistics.py
    │   draw_figure.py
    │   matrix.py
    │   similarity.py
    │   tools.py
    │   util.py
```


## Parameters Settings
If you want to change the default hyparameters, you can set it in `configx.py`. The meanings of the hyparameters is as follows:

#### Dataset Parameters

`dataset_name`: the short name of dataset, the default value is `ft`.

`k_fold_num`: the num of cross validation, the default value is `5`.

`rating_path `: the path of raw ratings data file, the default value is `../data/ft_ratings.txt`.

`rating_cv_path`: the cross validation path of ratings data, the default value is `../data/cv/`.

`trust_path`: the path of raw trust data file, the default value is `../data/ft_trust.txt`.

`sep`: the separator of rating and trust data in triple tuple, the default value is ` `.

`random_state`: the seed of random number, the default value is `0`.

`size`: the ratio of train set, the default value is `0.8`.

`min_val`: the minimum rating value, the default value is `0.5`.

`max_val`: the maximum rating value, the default value is `4.0`.

#### Model HyperParameters

`coldUserRating`: the number of ratings a cold start user rated on items, the default value is `5`.

`factor`: the size of latent dimension for user and item, the default value is `10`.

`threshold`: the threshold value of model training, the default value is `1e-4`.

`lr`: the learning rate, the default value is `0.01`.

`maxIter`: the maximum number of iterations, the default value is `100`.

`lambdaP`: the parameter of user regularizer, the default value is `0.001`.

`lambdaQ`: the parameter of item regularizer, the default value is `0.001`.

`gamma`: momentum coefficient, the default value is `0.9`.

`isEarlyStopping`: early stopping flag, the default value is `false`.

#### Output Parameters

`result_path`: the main directory of results, the default value is `../results/`.

`model_path`: the directory of well-trained variables, the default value is `../results/model/`.

`result_log_path`: the directory of logs when training models, the default value is `../results/log/`.

## Usage

Next, I will take `pmf` as an example to introduce how to execute our code.

First, we should split our rating data into several parts for training, testing and cross validation.
```
from utility.cross_validation import split_5_folds
from configx.configx import ConfigX

if __name__ == "__main__":
    configx = ConfigX()
    configx.k_fold_num = 5 
    configx.rating_path = "../data/ft_ratings.txt"
    configx.rating_cv_path = "../data/cv/"
    
    split_5_folds(configx)
```

Next, we open the `pmf.py` file in `model` folder, and configure the hyperparameters for training and execute the following code：

```
if __name__ == '__main__':

    rmses = []
    maes = []
    bmf = FunkSVDwithR()
    for i in range(bmf.config.k_fold_num):
        bmf.train_model(i)
        rmse, mae = bmf.predict_model()
        print("current best rmse is %0.5f, mae is %0.5f" % (rmse, mae))
        rmses.append(rmse)
        maes.append(mae)
    rmse_avg = sum(rmses) / 5
    mae_avg = sum(maes) / 5
    print("the rmses are %s" % rmses)
    print("the maes are %s" % maes)
    print("the average of rmses is %s " % rmse_avg)
    print("the average of maes is %s " % mae_avg)







