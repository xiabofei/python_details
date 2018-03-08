from data_split import TOXIC, SEVERE_TOXIC, OBSCENE, THREAT, INSULT, IDENTITY_HATE

params_groups = {
    TOXIC:
        {
            'objective': 'binary',
            'learning_rate': 0.1,
            'num_leaves':31,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'nthread': 8,
            'verbose': -1,
            'verbosity': -1,
            'min_split_gain': 0.1,
            'reg_alpha': 0.1,
        },
    SEVERE_TOXIC:
        {
            'objective': 'binary',
            'learning_rate': 0.1,
            'num_leaves':31,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'nthread': 8,
            'verbose': -1,
            'verbosity': -1,
            'min_split_gain': 0.1,
            'reg_alpha': 0.1,
        },
    OBSCENE:
        {
            'objective': 'binary',
            'learning_rate': 0.1,
            'num_leaves':31,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'nthread': 8,
            'verbose': -1,
            'verbosity': -1,
            'min_split_gain': 0.1,
            'reg_alpha': 0.1,
        },
    THREAT:
        {
            'objective': 'binary',
            'learning_rate': 0.1,
            'num_leaves':31,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'nthread': 8,
            'verbose': -1,
            'verbosity': -1,
            'min_split_gain': 0.1,
            'reg_alpha': 0.1,
        },
    INSULT:
        {
            'objective': 'binary',
            'learning_rate': 0.1,
            'num_leaves':31,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'nthread': 8,
            'verbose': -1,
            'verbosity': -1,
            'min_split_gain': 0.1,
            'reg_alpha': 0.1,
        },
    IDENTITY_HATE:
        {
            'objective': 'binary',
            'learning_rate': 0.1,
            'num_leaves':31,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'nthread': 8,
            'verbose': -1,
            'verbosity': -1,
            'min_split_gain': 0.1,
            'reg_alpha': 0.1,
        },
}
