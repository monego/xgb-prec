import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


autumn_data = pd.read_feather('data/autumn_data.feather')
winter_data = pd.read_feather('data/winter_data.feather')
spring_data = pd.read_feather('data/spring_data.feather')
summer_data = pd.read_feather('data/summer_data.feather')


def prepare_data(df, data_season, target_season):

    scaler = MinMaxScaler()
    scaler.fit(df[[target_season]])

    df = df[df.year >= 2017]
    prec = df.loc[:, target_season]
    df = df.loc[:, "year":"lon"]

    df[target_season] = scaler.transform(prec.values.reshape(-1, 1))

    df['prec_XGB'] = np.load("xgboost-output-" + target_season + ".npy")
    norm_prec = df[[target_season]]

    df[target_season + "_denorm"] = scaler.inverse_transform(norm_prec.values.reshape(-1, 1))
    norm_prec_XGB = df["prec_XGB"]

    df["prec_XGB_denorm"] = scaler.inverse_transform(norm_prec_XGB.values.reshape(-1, 1))

    prec_XGB_denorm = df["prec_XGB_denorm"]
    df["Error denorm [" + target_season + " - XGB]"] = prec - prec_XGB_denorm
    return df


do = prepare_data(autumn_data, 'autumn', 'winter')
di = prepare_data(winter_data, 'winter', 'spring')
dp = prepare_data(spring_data, 'spring', 'summer')
dv = prepare_data(summer_data, 'summer', 'autumn')

do2018 = do[do.year == 2018]
di2018 = di[di.year == 2018]
dp2018 = dp[dp.year == 2018]
dv2018 = dv[dv.year.astype(int) == 2018]

do2019 = do[do.year == 2019]
di2019 = di[di.year == 2019]
dp2019 = dp[dp.year == 2019]
dv2019 = dv[dv.year.astype(int) == 2019]


def rmse(df):
    return np.sqrt(np.mean((df)**2))


me2018_art_tf = np.array([-0.12, -0.07, -1.18, -0.96])
me2019_art_tf = np.array([0.09, -0.02, -0.34, 1.25])
rmse2018_art_tf = np.array([7.63, 0.86, 8.96, 4.20])
rmse2019_art_tf = np.array([2.51, 1.40, 1.32, 5.27])

me_xgb_2018 = np.array([dp2018["Error denorm [summer - XGB]"].mean(),
                        dv2018["Error denorm [autumn - XGB]"].mean(),
                        do2018["Error denorm [winter - XGB]"].mean(),
                        di2018["Error denorm [spring - XGB]"].mean()])

me_xgb_2019 = np.array([dp2019["Error denorm [summer - XGB]"].mean(),
                        dv2019["Error denorm [autumn - XGB]"].mean(),
                        do2019["Error denorm [winter - XGB]"].mean(),
                        di2019["Error denorm [spring - XGB]"].mean()])

rmse_xgb_2018 = np.array([rmse(dp2018["Error denorm [summer - XGB]"]),
                          rmse(dv2018["Error denorm [autumn - XGB]"]),
                          rmse(do2018["Error denorm [winter - XGB]"]),
                          rmse(di2018["Error denorm [spring - XGB]"])])

rmse_xgb_2019 = np.array([rmse(dp2019["Error denorm [summer - XGB]"]),
                          rmse(dv2019["Error denorm [autumn - XGB]"]),
                          rmse(do2019["Error denorm [winter - XGB]"]),
                          rmse(di2019["Error denorm [spring - XGB]"])])


# "TF" refers to TensorFlow. It's a comparison with DOI: 10.3390/rs13132468.

print("Mean error \t Spring->Summer 2018: \t", me_xgb_2018[0], "\t TF: \t", me2018_art_tf[0])
print("Mean error \t Summer->Autumn 2018: \t\t", me_xgb_2018[1], "\t TF: \t", me2018_art_tf[1])
print("Mean error \t Autumn->Winter 2018: \t\t", me_xgb_2018[2], "\t TF: \t", me2018_art_tf[2])
print("Mean error \t Winter->Spring 2018: \t", me_xgb_2018[3], "\t TF: \t", me2018_art_tf[3])

print("Mean error \t Spring->Summer 2019: \t", me_xgb_2019[0], "\t TF: \t", me2019_art_tf[0])
print("Mean error \t Summer->Autumn 2019: \t\t", me_xgb_2019[1], "\t TF: \t", me2019_art_tf[1])
print("Mean error \t Autumn->Winter 2019: \t\t", me_xgb_2019[2], "\t TF: \t", me2019_art_tf[2])
print("Mean error \t Winter->Spring 2019: \t", me_xgb_2019[3], "\t TF: \t", me2019_art_tf[3])

print("RMSE \t Spring->Summer 2018: \t", rmse_xgb_2018[0], "\t TF: \t", rmse2018_art_tf[0])
print("RMSE \t Summer->Autumn 2018: \t\t", rmse_xgb_2018[1], "\t TF: \t", rmse2018_art_tf[1])
print("RMSE \t Autumn->Winter 2018: \t\t", rmse_xgb_2018[2], "\t TF: \t", rmse2018_art_tf[2])
print("RMSE \t Winter->Spring 2018: \t", rmse_xgb_2018[3], "\t TF: \t", rmse2018_art_tf[3])

print("RMSE \t Spring->Summer 2019: \t", rmse_xgb_2019[0], "\t TF: \t", rmse2019_art_tf[0])
print("RMSE \t Summer->Autumn 2019: \t\t", rmse_xgb_2019[1], "\t TF: \t", rmse2019_art_tf[1])
print("RMSE \t Autumn->Winter 2019: \t\t", rmse_xgb_2019[2], "\t TF: \t", rmse2019_art_tf[2])
print("RMSE \t Winter->Spring 2019: \t", rmse_xgb_2019[3], "\t TF: \t", rmse2019_art_tf[3])

print("Difference ME [NN - XGB] (2018): ", me2018_art_tf - me_xgb_2018)
print("Difference ME [NN - XGB] (2019): ", me2019_art_tf - me_xgb_2019)
print("Difference RMSE [NN - XGB] (2018): ", rmse2018_art_tf - rmse_xgb_2018)
print("Difference RMSE [NN - XGB] (2019): ", rmse2019_art_tf - rmse_xgb_2019)

print("ME 2018 4 estações: ", np.mean(me2018_art_tf - me_xgb_2018))
print("ME 2019 4 estações: ", np.mean(me2019_art_tf - me_xgb_2019))
print("RMSE 2018 4 estações: ", np.mean(rmse2018_art_tf - rmse_xgb_2018))
print("RMSE 2019 4 estações: ", np.mean(rmse2019_art_tf - rmse_xgb_2019))

print("ME 2 anos: ", np.array([me2018_art_tf - me_xgb_2018,
                               me2019_art_tf - me_xgb_2019]).mean(axis=0))
print("RMSE 2 anos: ", np.array([rmse2018_art_tf - rmse_xgb_2018,
                                 rmse2019_art_tf - rmse_xgb_2019]).mean(axis=0))

do.to_excel("results/results-winter.xlsx")
di.to_excel("results/results-spring.xlsx")
dp.to_excel("results/results-summer.xlsx")
dv.to_excel("results/results-autumn.xlsx")
