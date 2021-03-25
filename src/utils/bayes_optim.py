from skopt import BayesSearchCV


def bayes_search(base_pipeline, search_list, X, y, cv):
    """Perform Bayes Search

    Args:
        base_pipeline (sklearn.pipeline.Pipeline): Base pipeline
        search_list (list of tuples): [(parameter space, # of evaluations),]
        X (np.array): data
        y (np.array): labels
        cv (str or sklearn.model_selection): Cross validation type

    Returns:
        skopt: fitted optimizer
    """

    opt = BayesSearchCV(
        base_pipeline,
        # (parameter space, # of evaluations)
        search_list,
        cv=cv)

    opt.fit(X, y)

    return opt
