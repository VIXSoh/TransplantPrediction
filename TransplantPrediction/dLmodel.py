import pandas as pd
import sidetable
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline 
from sklearn.preprocessing import StandardScaler

# set display options to show everything
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

# load dict file
data_dict = pd.read_csv("TransplantPrediction/Data/data_dictionary.csv")
print(data_dict.head(2))

# load data file
data = pd.read_csv("TransplantPrediction/Data/train.csv")

# Take a look at the data
print(data.info()) # 28800 entries, 60 columns
print(data.head(2))

# Check the structure of the data
print(data.columns)
print(data.shape)
data.describe()

# Check for duplicate data
data.loc[data.duplicated(),] # no duplicates

# divide into numeric data and categorical data
data_type = data.dtypes
data_type_object = data_type[data_type == "object"]
data_type_numeric = data_type[data_type != "object"]

data_object = data[data_type_object.index]
data_numeric = data[data_type_numeric.index]

# Refine categorical data
print(data_object.columns)
'''
Index(['dri_score', 'psych_disturb', 'cyto_score', 'diabetes', 'tbi_status',
       'arrhythmia', 'graft_type', 'vent_hist', 'renal_issue', 'pulm_severe',
       'prim_disease_hct', 'cmv_status', 'tce_imm_match', 'rituximab',
       'prod_type', 'cyto_score_detail', 'conditioning_intensity', 'ethnicity',
       'obesity', 'mrd_hct', 'in_vivo_tcd', 'tce_match', 'hepatic_severe',
       'prior_tumor', 'peptic_ulcer', 'gvhd_proph', 'rheum_issue', 'sex_match',
       'race_group', 'hepatic_mild', 'tce_div_match', 'donor_related',
       'melphalan_dose', 'cardiac', 'pulm_moderate'],
'''

# Check for null values 
for c in data_object.columns:
    print("======= %s =======" % c)
    print(data_object[c].value_counts(dropna=False))
    #print(data_object.stb.freq([c], cum_cols=False))


'''
======= dri_score =======
dri_score
Intermediate                                         10436
N/A - pediatric                                       4779
High                                                  4701
N/A - non-malignant indication                        2427
TBD cytogenetics                                      2003
Low                                                   1926
High - TED AML case <missing cytogenetics             1414
Intermediate - TED AML case <missing cytogenetics      481
N/A - disease not classifiable                         272
Very high                                              198
NaN                                                    154 - null value
Missing disease status                                   9 - null value
Name: count, dtype: int64

= > 163 null vlaues

======= psych_disturb =======
psych_disturb
No          23005
Yes          3587
NaN          2062 - null value
Not done      146 - null value
Name: count, dtype: int64

= > 2208 null values

======= cyto_score =======
cyto_score
Poor            8802
NaN             8068 - null value
Intermediate    6376
Favorable       3011
TBD             1341 - null value
Normal           643
Other            504
Not tested        55 - null value
Name: count, dtype: int64

= > 8123 null values

======= diabetes =======
diabetes
No          22201
Yes          4339
NaN          2119 - null value
Not done      141 - null value
Name: count, dtype: int64

= > 2260 null values

======= tbi_status =======
tbi_status
No TBI                              18861
TBI + Cy +- Other                    6104
TBI +- Other, <=cGy                  1727
TBI +- Other, >cGy                   1700
TBI +- Other, -cGy, single            134
TBI +- Other, -cGy, fractionated      119
TBI +- Other, -cGy, unknown dose       79
TBI +- Other, unknown dose             76
Name: count, dtype: int64

= > ?

======= arrhythmia =======
arrhythmia
No          25203
NaN          2202 - null values
Yes          1277
Not done      118 - null value
Name: count, dtype: int64

= > 2320 null values

======= graft_type =======
graft_type
Peripheral blood    20546
Bone marrow          8254
Name: count, dtype: int64

no null values

======= vent_hist =======
vent_hist
No     27721
Yes      820
NaN      259 - null value
Name: count, dtype: int64

= > 259 null values

======= renal_issue =======
renal_issue
No          26548
NaN          1915 - null value
Yes           200
Not done      137 -  null value
Name: count, dtype: int64

= > 2052

======= pulm_severe =======
pulm_severe
No          24779
NaN          2135 - null value
Yes          1706
Not done      180 - null value
Name: count, dtype: int64

= > 2315

======= prim_disease_hct =======
prim_disease_hct
ALL                     8102
AML                     7135
MDS                     3046
IPA                     1719
MPN                     1656
IEA                     1449
NHL                     1319
IIS                     1024
PCD                      869
SAA                      713
AI                       449
HIS                      445
Other leukemia           366
Solid tumor              207
IMD                      144
Other acute leukemia      83
HD                        54
CML                       20
Name: count, dtype: int64

no null values
======= cmv_status =======
cmv_status
+/+    13596
-/+     7081
+/-     4048
-/-     3441
NaN      634 - null value
Name: count, dtype: int64
======= tce_imm_match =======
tce_imm_match
P/P    13114
NaN    11133 - null value
G/G     2522
H/H     1084
G/B      544
H/B      229
P/H       83
P/B       66
P/G       25
Name: count, dtype: int64
======= rituximab =======
rituximab
No     26033
NaN     2148 - null value
Yes      619
Name: count, dtype: int64
======= prod_type =======
prod_type
PB    20381
BM     8419
Name: count, dtype: int64
======= cyto_score_detail =======
cyto_score_detail
NaN             11923 - null value
Intermediate    11158
Poor             3323
Favorable        1208
TBD              1043 - null value
Not tested        145 - null value
Name: count, dtype: int64
======= conditioning_intensity =======
conditioning_intensity
MAC                              12288
RIC                               7722
NaN                               4789 - null value
NMA                               3479
TBD                                373 - null value
No drugs reported                   87
N/A, F(pre-TED) not submitted       62 - null value
Name: count, dtype: int64
======= ethnicity =======
ethnicity
Not Hispanic or Latino      24482
Hispanic or Latino           3347
NaN                           587 - null value
Non-resident of the U.S.      384 - null value
Name: count, dtype: int64
======= obesity =======
obesity
No          25144
Yes          1779
NaN          1760 - null value
Not done      117 - null value
Name: count, dtype: int64
======= mrd_hct =======
mrd_hct
NaN         16597 - null value
Negative     8068
Positive     4135
Name: count, dtype: int64
======= in_vivo_tcd =======
in_vivo_tcd
No     17591
Yes    10984
NaN      225 - null value
Name: count, dtype: int64
======= tce_match =======
tce_match
NaN                   18996 - null value
Permissive             6272
GvH non-permissive     1605
Fully matched          1059
HvG non-permissive      868
Name: count, dtype: int64
======= hepatic_severe =======
hepatic_severe
No          25238
NaN          1871 - null value
Yes          1481
Not done      210 - null value
Name: count, dtype: int64
======= prior_tumor =======
prior_tumor
No          23828
Yes          3009
NaN          1678 - null value
Not done      285 - null value
Name: count, dtype: int64
======= peptic_ulcer =======
peptic_ulcer
No          25956
NaN          2419 - null value
Yes           259
Not done      166 - null value
Name: count, dtype: int64
======= gvhd_proph =======
gvhd_proph
FK+ MMF +- others                  10440
Cyclophosphamide alone              5270
FK+ MTX +- others(not MMF)          4262
Cyclophosphamide +- others          2369
CSA + MMF +- others(not FK)         2278
FKalone                             1230
Other GVHD Prophylaxis               550
TDEPLETION alone                     545
TDEPLETION +- other                  539
No GvHD Prophylaxis                  262
CDselect alone                       251
NaN                                  225 - null value
CSA + MTX +- others(not MMF,FK)      224
CSA alone                            214
Parent Q = yes, but no agent          62
CDselect +- other                     55
CSA +- others(not FK,MMF,MTX)         23
FK+- others(not MMF,MTX)               1
Name: count, dtype: int64
======= rheum_issue =======
rheum_issue
No          26015
NaN          2183 - null value
Yes           457
Not done      145 - null value
Name: count, dtype: int64
======= sex_match =======
sex_match
M-M    7980
F-M    7822
M-F    6715
F-F    6022
NaN     261
Name: count, dtype: int64
======= race_group =======
race_group
More than one race                           4845
Asian                                        4832
White                                        4831
Black or African-American                    4795
American Indian or Alaska Native             4790
Native Hawaiian or other Pacific Islander    4707
Name: count, dtype: int64
======= hepatic_mild =======
hepatic_mild
No          24989
NaN          1917 - null value
Yes          1754
Not done      140 - null value
Name: count, dtype: int64
======= tce_div_match =======
tce_div_match
Permissive mismatched            12936
NaN                              11396 - null value
GvH non-permissive                2458
HvG non-permissive                1417
Bi-directional non-permissive      593
Name: count, dtype: int64
======= donor_related =======
donor_related
Related                     16208
Unrelated                   12088
Multiple donor (non-UCB)      346
NaN                           158 - null value
Name: count, dtype: int64
======= melphalan_dose =======
melphalan_dose
N/A, Mel not given    20135
MEL                    7260
NaN                    1405 - null value
Name: count, dtype: int64
======= cardiac =======
cardiac
No          24592
NaN          2542 - null value
Yes          1519
Not done      147 - null value
Name: count, dtype: int64
======= pulm_moderate =======
pulm_moderate
No          21338
Yes          5249
NaN          2047 - null value
Not done      166 - null value
Name: count, dtype: int64
'''



# load without na values 
na_vals = ["TBD cytogenetics","Not done","NaN","Non-resident of the U.S.","TBD","N/A, F(pre-TED) not submitted","Not tested","Missing disease status"]

# load data file
data = pd.read_csv("TransplantPrediction/Data/train.csv", na_values=na_vals).drop(columns=['ID'])

# Find how many na values each columns have and drop problematic ones.
threshold = 6000 # ~ 20 percent of the data
drop_index = []

for c in data.columns:
    if (data[c].isna().sum() > threshold):
        print("======= %s =======" % c)
        print(data[c].isna().value_counts(dropna=False))
        drop_index.append(c)

data = data.drop(columns=drop_index)

# Find how many na values each rows have 
na_row_counts = data.isna().sum(axis=1)
na_row_counts.value_counts()
# drop rows with more than 5 na values
data = data[na_row_counts <= 5]
prediction = data['efs']
data = data.drop(columns=['efs'])

# divide into numeric data and categorical data
data_type = data.dtypes
data_type_object = data_type[data_type == "object"]
data_type_numeric = data_type[data_type != "object"]

data_object = data[data_type_object.index]
data_numeric = data[data_type_numeric.index]

# Check each predictors to see if they are categorical values stored with numeric types
for c in data_numeric.columns:
    print("======= %s =======" % c)
    print(data_numeric[c].describe())
# Actual numeric data
# efs_time karnofsky_score comorbidity_score age_at_hct donor_age

true_numeric_index  =['efs_time', 'karnofsky_score', 'comorbidity_score', 'age_at_hct', 'donor_age']
true_categorical_index = data.drop(columns=true_numeric_index).columns

# separate categorical predictors that are stored as numeric values
data_numeric_cat = data_numeric.drop(columns=true_numeric_index)
data_numeric = data_numeric[true_numeric_index]

# create transformers
numeric_transformer = Pipeline(steps=[('imputer', KNNImputer(n_neighbors=3))])
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

knn_imputer = ColumnTransformer(
    transformers=[
('num', numeric_transformer, true_numeric_index),
('cat', categorical_transformer, true_categorical_index)
    ]
)

# need more work but impute and convert back to df
df_imputed = knn_imputer.fit_transform(data)

# convert back to DataFrame
ohe = knn_imputer.named_transformers_['cat'].named_steps['onehot']
cat_feature_names = ohe.get_feature_names_out(true_categorical_index)
all_feature_names = true_numeric_index + list(cat_feature_names)

# Create final DataFrame
df_final = pd.DataFrame.sparse.from_spmatrix(df_imputed, columns=all_feature_names)

# Find how many na values each columns have
for c in df_final.columns:
    # if (df_final[c].isna().sum() > threshold):
    print("======= %s =======" % c)
    print(df_final[c].isna().value_counts(dropna=False))


corr_matrix = data_numeric.corr()

plt.figure(figsize=(10, 8)) 
sns.heatmap(corr_matrix, annot=True, cmap ='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Matrix')
plt.show()

df = pd.DataFrame(data_numeric)

sns.pairplot(df)
plt.suptitle('Pairplots of numerics', y=1.02)
plt.show()

# Loop through columns and create plots
for col in data_numeric.columns:
    plt.figure()  # Create a new figure for each plot
    data_numeric[col].plot(kind='scatter')  # Change 'line' to 'bar', 'hist', etc. as needed
    plt.title(col)
    plt.show()

data_numeric['hla_match_c_high'].isna().value_counts(dropna=False)

'''
'ID', 'hla_match_c_high', 'hla_high_res_8', 'hla_low_res_6',
       'hla_high_res_6', 'hla_high_res_10', 'hla_match_dqb1_high',
       'hla_nmdp_6', 'hla_match_c_low', 'hla_match_drb1_low',
       'hla_match_dqb1_low', 'year_hct', 'hla_match_a_high', 'donor_age',
       'hla_match_b_low', 'age_at_hct', 'hla_match_a_low', 'hla_match_b_high',
       'comorbidity_score', 'karnofsky_score', 'hla_low_res_8',
       'hla_match_drb1_high', 'hla_low_res_10', 'efs', 'efs_time'],
      dtype='object'
'''
#prediction = data['efs']




# could plot for small dataset.
# plt.figure(figsize=(10, 8)) 
# sns.heatmap(corr_matrix, annot=True, cmap ='coolwarm', fmt='.2f', linewidths=0.5)
# plt.title('Correlation Matrix')
# plt.show()

# df = pd.DataFrame(data_numeric)

# sns.pairplot(df)
# plt.suptitle('Pairplots of numerics', y=1.02)
# plt.show()

# Loop through columns and create plots
for col in data_numeric.columns:
    plt.figure()  # Create a new figure for each plot
    data_numeric[col].plot(kind='scatter')  # Change 'line' to 'bar', 'hist', etc. as needed
    plt.title(col)
    plt.show()

data_numeric['hla_match_c_high'].isna().value_counts(dropna=False)

'''
'ID', 'hla_match_c_high', 'hla_high_res_8', 'hla_low_res_6',
       'hla_high_res_6', 'hla_high_res_10', 'hla_match_dqb1_high',
       'hla_nmdp_6', 'hla_match_c_low', 'hla_match_drb1_low',
       'hla_match_dqb1_low', 'year_hct', 'hla_match_a_high', 'donor_age',
       'hla_match_b_low', 'age_at_hct', 'hla_match_a_low', 'hla_match_b_high',
       'comorbidity_score', 'karnofsky_score', 'hla_low_res_8',
       'hla_match_drb1_high', 'hla_low_res_10', 'efs', 'efs_time'],
      dtype='object'
'''
#prediction = data['efs']



# Check for high correlations. 
corr_matrix = df_final.corr()

mask = np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    
# Apply the mask
masked_corr_matrix = corr_matrix.where(mask)

# Stack the DataFrame to get a Series and drop masked values
stacked_corr = masked_corr_matrix.stack().dropna()

# Filter for correlations that meet the threshold
high_corr_pairs = stacked_corr[abs(stacked_corr) > 0.8]

# Sort the results in descending order by absolute correlation value
high_corr_pairs = high_corr_pairs.sort_values(ascending=False, key=abs)

# Format the output
print(f"Feature pairs with correlation magnitude > 0.8:")
result_list = []
for pair, value in high_corr_pairs.items():
    feature1, feature2 = pair
    result_list.append((feature1, feature2, value))
    print(f"* {feature1} and {feature2}: {value:.4f}")

print(result_list)

# 
