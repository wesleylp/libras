import pandas as pd


def df_results(opt):
    df = pd.DataFrame(opt.cv_results_['params'])
    # df.rename(columns = {0:'param_model'}, inplace = True)

    df_mean = pd.DataFrame(opt.cv_results_['mean_test_score'])
    df_std = pd.DataFrame(opt.cv_results_['std_test_score'])
    df_rank = pd.DataFrame(opt.cv_results_['rank_test_score'])

    df = df.join(df_mean)
    df.rename(columns={0: 'mean_test_score'}, inplace=True)

    df = df.join(df_std)
    df.rename(columns={0: 'std_test_score'}, inplace=True)

    df = df.join(df_rank)
    df.rename(columns={0: 'rank'}, inplace=True)

    df.sort_values(by='mean_test_score', inplace=True, ascending=False)

    return df
