import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cleaner
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder

dataset = pd.read_csv("data/properties.csv")

# Step 1: Clean the dataset by removing columns with too many missing values
cleaned_df = cleaner.clean_dataset(dataset)

# Step 2: Replace NaN values with mean values for specific columns
cleaned_df = cleaner.remove_NaN(cleaned_df)

#manualy clean DataFrame more
cleaned_df.drop(['subproperty_type', 'region', 'latitude', 'longitude', 'fl_furnished', 'fl_floodzone', 'fl_double_glazing', 'fl_terrace', 'fl_garden' ], axis=1, inplace=True)

obj = (cleaned_df.dtypes == 'object')
object_cols = list(obj[obj].index)
print("Categorical variables:",len(object_cols))

int_ = (cleaned_df.dtypes == 'int')
num_cols = list(int_[int_].index)
print("Integer variables:",len(num_cols))

fl = (cleaned_df.dtypes == 'float')
fl_cols = list(fl[fl].index)
print("Float variables:",len(fl_cols))

# Heatmap Seaborn
# Select only numerical features for correlation analysis
numerical_dataset = cleaned_df.select_dtypes(include=['number'])

plt.figure(figsize=(12, 6))
sns.heatmap(numerical_dataset.corr(),
            cmap = 'BrBG',
            fmt = '.2f',
            linewidths = 2,
            annot = True)

unique_values = []
for col in object_cols:
  unique_values.append(cleaned_df[col].unique().size)
plt.figure(figsize=(10,6))
plt.title('No. Unique values of Categorical Features')
plt.xticks(rotation=90)
sns.barplot(x=object_cols,y=unique_values)


plt.figure(figsize=(18, 36))
plt.title('Categorical Features: Distribution')
plt.xticks(rotation=90)
index = 1

for col in object_cols:
    y = cleaned_df[col].value_counts()
    plt.subplot(11, 4, index)
    plt.xticks(rotation=90)
    sns.barplot(x=list(y.index), y=y)
    index += 1


#plt.show()
