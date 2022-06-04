import os

import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm
from sklearn import metrics
from feature_engineering import feature_analysor

# inputpath = '/Users/xiechunyao/Downloads/tabular-playground-series-may-2022'
# for dirname, _, filenames in os.walk(inputpath):
#     for filename in filenames:
#         print(os.path.join(dirname,filename))

train = pd.read_csv('/Users/xiechunyao/Downloads/tabular-playground-series-may-2022/train.csv')
test = pd.read_csv('/Users/xiechunyao/Downloads/tabular-playground-series-may-2022/test.csv')
float_features = [f for f in train.columns if train[f].dtype == 'float64']
int_features = [f for f in test.columns if test[f].dtype == 'int64' and f != 'id']
f_analysor = feature_analysor(train, test)

# The target is binary. With 51 % zeros and 49 % ones, it is almost balanced
# f_analysor.target_portion()

# The histograms of the 16 float features show that all these features are normally distributed
# with the center at zero. f_00 through f_06 have a standard deviation of 1; f_19 through f_26
# have a standard deviation between 2.3 and 2.5, and f_28 has a standard deviation of almost 240.
# Training and test data have the same distribution. There seem to be no outliers:
# f_analysor.show_float_feature_distribution(float_features)

# The correlation matrix shows:
#
# 1 f_00 through f_06 are correlated with f_28, but not with each other.
# 2 f_19 through f_26 are all slightly correlated with each other.
# 3 No feature is strongly correlated with the target.
# f_analysor.correlation_of_float_feature(float_features)

# The correlation matrix shows only linear dependences.
# If we plot a rolling mean of the target probability for every feature, we'll see nonlinear dependences as well.
# A horizontal line means that the target does not depend on the feature (e.g., f_03, f_04, f_06),
# a line with low minimum and high maximum shows a high mutual information between feature and target (e.g., f_19, f_21, f_28).
# f_analysor.plot_mutual_info_diagram(float_features)

# Looking at the histograms of the integer features, we see that the first twelve features all have values between 0 and 16.
# The last two features are special: f_29 is binary and f_30 is ternary.
# f_analysor.integer_features(int_features)
# f_analysor.plot_mutual_info_diagram(int_features, title='How the target probability depends on the int features')

# f_27 is a string feature which cannot be used as a feature as-is. Let's find out how to engineer something useful from it!
#
# We first verify that the string always has length 10:
# f_analysor.string_feature_size()

# The 900000 train samples contain 741354 different values, i.e. most of the strings are different.
# The most frequent string, 'BBBBBBCJBC' occurs only 12 times:
# f_analysor.count_string_feature_values()

# It is important to understand whether the f_27 strings in test are the same as in training.
# Unfortunately, test contains 1181880 - 741354 = 440526 strings which do not occur in training.
#
# Insight: We must not use this string as a categorical feature in a classifier.
# Otherwise, the model learns to rely on strings which never occur in the test data.
# f_analysor.count_strin_feature_values_all_dataset()


# In the next step, we look at the distributions of the letters at every one of the ten positions in the string.
# We see that positions 0, 2 and 5 are binary; the other positions have more values.
# Every position gives some information about the target (the target means depend on the feature value).
# f_analysor.see_str_pos()

# We can as well count the unique characters in the string and use this count as a feature
# f_analysor.see_unique_chr()

# You can use the following lines of code to split the strings into ten numerical features:
# f_analysor.string_feature_transformer()
str_feature = [f for f in train.columns if f.startswith('ch')] + ['unique_characters']
# f_analysor.plot_mutual_info_diagram(str_feature,title='How the target probability depends on the character features')

# Top three feature interactions
# the projection to f_02 and f_21
# the projection to f_05 and f_22
# the projection to f_00+f_01 and f_26
f_analysor.twoD_feature()

# We now can either hope that our classifier finds these borders by itself, or we can help the classifier.
#
# And how can we help a classifier? For every projection, we create a ternary categorical feature that indicates to which region a sample belongs:
#
# Top right region (high probability of target == 1) → +1
# Middle region (medium probability of target == 1) → 0
# Bottom left region (low probability of target == 1) → -1
# You can use the following lines of code to add the three features to the dataframes:

