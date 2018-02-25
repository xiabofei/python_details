from data_split import TOXIC, SEVERE_TOXIC, OBSCENE, THREAT, INSULT, IDENTITY_HATE

params_groups = {
    TOXIC:
        {
            'objective': 'binary', 'learning_rate': 0.1,
            'feature_fraction': 0.8, 'bagging_fraction': 0.8,
            'bagging_freq': 10, 'max_bin': 255,
            'nthread': 12, 'verbose': 0,
        },
    SEVERE_TOXIC:
        {
            'objective': 'binary', 'learning_rate': 0.1,
            'feature_fraction': 0.8, 'bagging_fraction': 0.8,
            'bagging_freq': 10, 'max_bin': 255,
            'nthread': 12, 'verbose': 0,
        },
    OBSCENE:
        {
            'objective': 'binary', 'learning_rate': 0.1,
            'feature_fraction': 0.8, 'bagging_fraction': 0.8,
            'bagging_freq': 10, 'max_bin': 255,
            'nthread': 12, 'verbose': 0,
        },
    THREAT:
        {
            'objective': 'binary', 'learning_rate': 0.1,
            'feature_fraction': 0.8, 'bagging_fraction': 0.8,
            'bagging_freq': 10, 'max_bin': 255,
            'nthread': 12, 'verbose': 0,
        },
    INSULT:
        {
            'objective': 'binary', 'learning_rate': 0.1,
            'feature_fraction': 0.8, 'bagging_fraction': 0.8,
            'bagging_freq': 10, 'max_bin': 255,
            'nthread': 12, 'verbose': 0,
        },
    IDENTITY_HATE:
        {
            'objective': 'binary', 'learning_rate': 0.1,
            'feature_fraction': 0.8, 'bagging_fraction': 0.8,
            'bagging_freq': 10, 'max_bin': 255,
            'nthread': 12, 'verbose': 0,
        },
}
