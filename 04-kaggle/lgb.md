```python
model = lgb.train(params,
                      train_set=d_train,
                      num_boost_round=10000,
                      valid_sets=watchlist,
                      verbose_eval=verbose_eval,
                      callbacks=[lgb.reset_parameter(learning_rate=[0.005]*1000+[0.003]*1000+[0.001]*8000)],
                      early_stopping_rounds=early_stop)
```

