import numpy as np
import pandas as pd
from scipy.stats import pearsonr, chi2_contingency
import matplotlib.pyplot as plt
import seaborn as sns

import codecademylib3
np.set_printoptions(suppress=True, precision = 2)

nba = pd.read_csv('./nba_games.csv')

# Subset Data to 2010 Season, 2014 Season
nba_2010 = nba[nba.year_id == 2010]
nba_2014 = nba[nba.year_id == 2014]

print(nba_2010.head())
print(nba_2014.head())

# 1
knicks_pts_10 =nba_2010.pts[nba_2010.fran_id == "Knicks"]
nets_pts_10 =nba_2010.pts[nba_2010.fran_id == "Nets"]

# 2 diffMean = 9.731707317073173 -Highly associated variables tend to have a large mean or median difference

mean_knicks = np.mean(knicks_pts_10)
mean_nets = np.mean(nets_pts_10)
diff_means_2010 = mean_knicks - mean_nets
print(diff_means_2010)

# 3 Cover almost the same area but Knicks are slightly more right leaning

plt.hist(knicks_pts_10,label = "Knicks",normed = True, alpha = 0.8)
plt.hist(nets_pts_10,label = "Nets",normed = True, alpha = 0.8)
plt.legend()
plt.title("2010")
plt.show()
plt.clf()

#4a 2014

knicks_pts_14 =nba_2014.pts[nba_2014.fran_id == "Knicks"]
nets_pts_14 =nba_2014.pts[nba_2014.fran_id == "Nets"]

#  4b diffMean = 0.44706798131809933 -Less associated variables tend to have a tiny mean or median difference

mean_knicks = np.mean(knicks_pts_14)
mean_nets = np.mean(nets_pts_14)
diff_means_2014 = mean_knicks - mean_nets
print(diff_means_2014)

# 4c Cover almost the same area but Knicks are slightly more left leaning

plt.hist(knicks_pts_14,label = "Knicks",normed = True, alpha = 0.8)
plt.hist(nets_pts_14,label = "Nets",normed = True, alpha = 0.8)
plt.legend()
plt.title("2014")
plt.show()
plt.clf()

#5 Thunder, Spurs, Knick Overlap while Celtic and Nets don't have that much overlap ->  First - Stronger Association, Last two less strong association

sns.boxplot(data = nba_2010, x = "fran_id", y= "pts")
plt.show()
plt.clf()

#6 Teams tend to lose less at Home then Away

location_result_freq = pd.crosstab(nba_2010.game_result,nba_2010.game_location)
print(location_result_freq)

#7
location_result_proportions = location_result_freq / len(nba_2010)
print(location_result_proportions)

#8 Chi-Square statistic larger than around 4 would strongly suggest an association between the variables

from scipy.stats import chi2_contingency
chi2,pval,dof,expected =chi2_contingency(location_result_freq)
print(np.round(expected))

chi2,pval,dof,expected =chi2_contingency(location_result_freq)
print(np.round(chi2))

#10 CA covariance of 0 indicates no linear relationship

point_diff_forecast_cov = np.cov(nba_2010.forecast,nba_2010.point_diff)
print(point_diff_forecast_cov)

#10 0.44020887084680815 = Moderate positive Correlation between between forecast and point_diff

from scipy.stats import pearsonr
point_diff_forecast_corr, p = pearsonr(nba_2010.forecast,nba_2010.point_diff)
print(point_diff_forecast_corr)

#11

plt.clf()
plt.scatter(x = nba_2010.forecast, y= nba_2010.point_diff)
plt.xlabel("Forecast")
plt.ylabel("Point_difference")
plt.show()

