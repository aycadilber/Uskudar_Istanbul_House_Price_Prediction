import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import missingno as msno
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)

df = pd.read_csv("hepsiemlak.csv")

########################## GENEL RESİM ########################################

def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.describe([0, 0.05, 0.50, 0.95, 0.99, 1]).T)

check_df(df)


def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]

    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")

    if na_name:
        return na_columns

missing_df = missing_values_table(df)

df.dtypes
df.shape  #(2856, 12)

####################################### SİLİNECEK SÜTUNLAR ##################################

columns_to_drop = ['Tapu Durumu', 'Takas', 'Krediye Uygunluk', 'Site İçerisinde', 'Aidat', 'Depozito', 'Cephe',
                   'Yetkili Ofis', "Yapının Durumu", 'İlan Durumu', 'Konut Şekli', 'Son Güncelleme Tarihi', 'İlan no']
df.drop(columns=columns_to_drop, axis=1, inplace=True)


################################# ODA SAYISI İÇİN DÜZENLEME #################################
# oda sayısı max değeri 380 ????

df[['Oda', 'Salon']] = df['Oda + Salon Sayısı'].str.split('+', expand=True).astype(int)
df["TOTALROOM"] = df["Oda"] + df["Salon"]
df.drop(columns=['Oda', 'Salon', 'Oda + Salon Sayısı'], inplace=True)
df.head(10)


################################# BRÜT / NET M2 İÇİN DÜZENLEME #################################

df["Brüt / Net M2"] = df["Brüt / Net M2"].str.extract(r"(\d+)").astype(int)

df["NETMETREKARE"] = df["Brüt / Net M2"]
df.drop("Brüt / Net M2", axis=1, inplace=True)


df.isnull().sum()
################################# KAT SAYISI İÇİN DÜZENLEME #################################
df['FLOOR'] = df['Kat Sayısı'].replace(['Katlı'], '', regex=True).astype(int)
df.drop("Kat Sayısı", axis=1, inplace=True)

df['FLOOR'].unique()


################################# BULUNDUĞU KAT İÇİN DÜZENLEME #################################

#df['Bulunduğu Kat'] = df['Bulunduğu Kat'].str.replace('\.', '', regex=True).str.replace('Kat', '', regex=True).str.replace(' ı', '', regex=True)
df['Bulunduğu Kat'] = df['Bulunduğu Kat'].str.replace('\.', '', regex=True).str.replace('Katı', '', regex=True).str.replace('Kat', '', regex=True)
#df['Bulunduğu Kat'] = df['Bulunduğu Kat'].str.replace(' Yüksek', '', regex=True).str.replace('Yarı', '', regex=True)
df['Bulunduğu Kat'] = df['Bulunduğu Kat'].str.replace('Yüksek Giriş', 'Yüksek', regex=True).str.replace('Yarı Bodrum', 'Bodrum', regex=True)


df['Bulunduğu Kat'] = df['Bulunduğu Kat'].str.replace(r'\bGiriş\b', '0', regex=True)
df['Bulunduğu Kat'] = df['Bulunduğu Kat'].str.replace(r'\bBahçe\b', '0', regex=True)
df['Bulunduğu Kat'] = df['Bulunduğu Kat'].str.replace(r'\bVilla\b', '0', regex=True)
df['Bulunduğu Kat'] = df['Bulunduğu Kat'].str.replace(r'\bBodrum\b', '0', regex=True)
df['Bulunduğu Kat'] = df['Bulunduğu Kat'].str.replace(r'\bZemin\b', '0', regex=True)
df['Bulunduğu Kat'] = df['Bulunduğu Kat'].str.replace(r'\bAra\b', '0.5', regex=True)
df['Bulunduğu Kat'] = df['Bulunduğu Kat'].str.replace(r'\bYüksek\b', '0.5', regex=True)


conditions = df['Bulunduğu Kat'].isin(['Teras ', 'En Üst ', 'Çatı '])
df.loc[conditions, 'Bulunduğu Kat'] = df.loc[conditions, 'FLOOR']

df['Bulunduğu Kat'] = df['Bulunduğu Kat'].astype(str)


df['Bulunduğu Kat'] = df['Bulunduğu Kat'].str.replace('Kot 1', '-1', regex=False)
df['Bulunduğu Kat'] = df['Bulunduğu Kat'].str.replace('Kot 2', '-2', regex=False)
df['Bulunduğu Kat'] = df['Bulunduğu Kat'].str.replace('Kot 3', '-3', regex=False)
df['Bulunduğu Kat'] = df['Bulunduğu Kat'].str.replace('21 ve üzeri', '33', regex=False)

kot_1_count = df[df['Bulunduğu Kat'] == 'Kot 1'].shape[0]
print("Kot 1 olanların sayısı:", kot_1_count)


df = df.rename(columns={'Bulunduğu Kat': 'FLOORNUMBER'})
df['FLOORNUMBER'] = pd.to_numeric(df['FLOORNUMBER'], errors='coerce')

df['FLOORNUMBER'].unique()

condition = df["FLOORNUMBER"] > df["FLOOR"]
df.loc[condition, ["FLOOR", "FLOORNUMBER"]] = df.loc[condition, ["FLOORNUMBER", "FLOOR"]].values



################################# PRICE İÇİN DÜZENLEME #################################

df['Price'] = df['Price'].str.replace('TL', '', regex=True)
df['PRICE'] = df['Price'].str.replace('.', '', regex=True).astype(int)

df.drop("Price", axis=1, inplace=True)


############################## BİNA YAŞI İÇİN DÜZENLEME #################################
df.head()
df['Bina Yaşı'].unique()
df['Bina Yaşı'] = df['Bina Yaşı'].replace(['Yaşında', 'Bina'], '', regex=True)
df['AGE'] = df['Bina Yaşı'].str.replace(r'\bSıfır\b', '0', regex=True)

df.drop("Bina Yaşı", axis=1, inplace=True)

df['AGE'] = pd.to_numeric(df['AGE'], errors='coerce')



############################ BANYO SAYISI ########################################
df['Banyo Sayısı'].value_counts()
df['Banyo Sayısı'].fillna(1, inplace=True)
df['BATHROOMS'] = df['Banyo Sayısı'].astype(int)
df.drop("Banyo Sayısı", axis=1, inplace=True)


df['Yakıt Tipi'].value_counts()
df = df.rename(columns={'Isınma Tipi': 'HEATINGSYSTEM'})
df = df.rename(columns={'Eşya Durumu': 'FURNISHED'})
df = df.rename(columns={'Yakıt Tipi': 'FUELTYPE'})
df = df.rename(columns={'Kullanım Durumu': 'USAGESTATUS'})
df = df.rename(columns={'Yapı Tipi': 'BUILDINGTYPE'})



# Find duplicate rows based on all columns
duplicate_rows = df[df.duplicated()]
print(duplicate_rows)

df = df.drop_duplicates(keep='first')


#  Outliers

def outlier_thresholds(dataframe, col_name, q1=0.05, q3=0.95):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


def grab_col_names(dataframe, cat_th=10, car_th=20):
    """

    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

    Parameters
    ------
        dataframe: dataframe
                Değişken isimleri alınmak istenilen dataframe
        cat_th: int, optional
                numerik fakat kategorik olan değişkenler için sınıf eşik değeri
        car_th: int, optinal
                kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    ------
        cat_cols: list
                Kategorik değişken listesi
        num_cols: list
                Numerik değişken listesi
        cat_but_car: list
                Kategorik görünümlü kardinal değişken listesi

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols'un içerisinde.
        Return olan 3 liste toplamı toplam değişken sayısına eşittir: catCols + numCols + catButCar = değişken sayısı

    """

    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                 dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                 dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'catCols: {len(cat_cols)}')
    print(f'numCols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'numButCat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car

cat_cols, num_cols, cat_but_car = grab_col_names(df)
num_cols.append("BATHROOMS")

for col in num_cols:
    plt.title(col)
    sns.boxplot(df[col])
    plt.show(block=True)


def check_outlier(dataframe, col_name, q1=0.05, q3=0.95):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name, q1, q3)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

for col in num_cols:
    print(col, check_outlier(df, col))



def remove_outlier(dataframe, col_name, q1=0.05, q3=0.95):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name, q1, q3)
    df_without_outliers = dataframe[~((dataframe[col_name] < low_limit) | (dataframe[col_name] > up_limit))]
    return df_without_outliers


def replace_with_thresholds(dataframe, variable, q1=0.05, q3=0.95):
    low_limit, up_limit = outlier_thresholds(dataframe, variable, q1, q3)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit



df[df["FLOORNUMBER"] == 33]
df = df.drop(787)
df = df.drop(1856)

df["AGE"].describe([0.01, 0.05, 0.25, 0.75, 0.90, 0.95, 0.99]).T
outlier_thresholds(df, "AGE", q1=0.25, q3=0.75)
replace_with_thresholds(df, "AGE", q1=0.25, q3=0.75)

df["TOTALROOM"].describe([0.01, 0.05, 0.25, 0.75, 0.90, 0.95, 0.99])
outlier_thresholds(df, "TOTALROOM", q1=0.1, q3=0.9)
replace_with_thresholds(df, "TOTALROOM", q1=0.1, q3=0.9)

df["FLOORNUMBER"].describe([0.01, 0.05, 0.25, 0.75, 0.90, 0.95, 0.99])
outlier_thresholds(df, "FLOORNUMBER")
replace_with_thresholds(df, "FLOORNUMBER")

df["BATHROOMS"].describe([0.01, 0.05, 0.25, 0.75, 0.90, 0.95, 0.99])
outlier_thresholds(df, "BATHROOMS")
replace_with_thresholds(df, "BATHROOMS")

df["PRICE"].describe([0.01, 0.05, 0.25, 0.75, 0.90, 0.95, 0.99])
outlier_thresholds(df, "PRICE")
replace_with_thresholds(df, "PRICE")

df["FLOOR"].describe([0.01, 0.05, 0.25, 0.75, 0.90, 0.95, 0.99])
outlier_thresholds(df, "FLOOR")
replace_with_thresholds(df, "FLOOR")

df["NETMETREKARE"].describe([0.01, 0.05, 0.25, 0.75, 0.90, 0.95, 0.99])
outlier_thresholds(df, "NETMETREKARE")
replace_with_thresholds(df, "NETMETREKARE")



for col in num_cols:
    print(col, check_outlier(df, col))


sns.boxplot(data=df["NETMETREKARE"])
plt.show()

sns.boxplot(data=df["AGE"])
plt.show()


sns.displot(data=df["PRICE"])
plt.show()


###################################
def plot_boxplots(dataframe, num_cols, ncols=2):
    nrows = (len(num_cols) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, 5 * nrows))

    colors = sns.color_palette('Set3', n_colors=len(dataframe.columns))

    for i, col in enumerate(num_cols):
        if col in dataframe.columns:
            ax = axes[i // ncols, i % ncols]
            sns.boxplot(x=dataframe[col], ax=ax, color=colors[i])
            ax.set_title(f"Box Plot of {col}")

    plt.tight_layout()
    plt.show()


plot_boxplots(df, num_cols)


def hist_plot(dataframe, num_cols, ncols=2, bins=10):
    nrows = (len(num_cols) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, 5 * nrows))

    for i, col in enumerate(num_cols):
        if col in dataframe.columns:
            ax = axes[i // ncols, i % ncols]
            sns.histplot(dataframe[col], ax=ax, bins=bins, kde=True)
            ax.set_title(f"Histogram of {col}")

    plt.tight_layout()
    plt.show()


hist_plot(df, num_cols)


######################################
# 2.Kategorik Değişken Analizi
######################################

# kategorik değişkenlerin sınıflarını ve bu sınıfların oranlarını getiren fonksiyon
def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")

    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show(block=True)

for col in cat_cols:
    cat_summary(df, col, plot=True)



######################################
# 3.Sayısal Değişken Analizi
######################################

def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)

for col in num_cols:
    num_summary(df, col, plot=True)


######################################
# 4.Hedef Değişken Analizi
######################################

def target_summary_with_cat(dataframe, target, categorical_col):
    print(pd.DataFrame({"TARGET_MEAN": dataframe.groupby(categorical_col)[target].mean()}), end="\n\n\n")

for col in cat_cols:
    target_summary_with_cat(df, "PRICE", col)


######################################
# 5.Korelasyon Analizi
######################################

def high_correlated_cols(dataframe, plot=False, corr_th=0.90):
    corr = dataframe.corr()
    cor_matrix = corr.abs()
    upper_triangle_matrix = cor_matrix.where(np.triu(np.ones(cor_matrix.shape), k=1).astype(np.bool_))
    drop_list = [col for col in upper_triangle_matrix.columns if any(upper_triangle_matrix[col] > corr_th)]
    if plot:
        import seaborn as sns
        import matplotlib.pyplot as plt
        sns.set(rc={'figure.figsize': (15, 15)})
        sns.heatmap(corr, cmap="RdBu", annot=True)
        plt.show(block=True)
    return drop_list

high_correlated_cols(df, plot=True)


###################################
# 6.Eksik Veri Analizi
###################################

df.isnull().sum()
def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]

    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")

    if na_name:
        return na_columns

missing_values_table(df, True)


##########################################
# Eksik Verinin Yapısını İncelemek
##########################################

# matrix: Değişkenlerdeki eksikliklerin birlikte cıkıp cıkmadıgı bilgisini verir
msno.matrix(df)
plt.show()

#heatmap : ısı haritası , eksiklikler birlikte mi cıkıyor bağımlıgınını anlayabilmek için kullanırız
#pozitif yönde korelasyon değişkenlerdeki eksikliklerin birlikte ortaya cıktıgı düşünülür .
msno.heatmap(df)

correlation = df['FUELTYPE'].fillna('Missing').groupby(df['BUILDINGTYPE']).value_counts(normalize=True)

#########################################
# Eksik Değer Problemini Çözme
#########################################

buildingtype_mode = df["BUILDINGTYPE"].mode()[0]
df["BUILDINGTYPE"].fillna(buildingtype_mode, inplace=True)

fueltype_mode = df["FUELTYPE"].mode()[0]
df["FUELTYPE"].fillna(fueltype_mode, inplace=True)

df["USAGESTATUS"].unique()
df["USAGESTATUS"].fillna('Belirtilmemiş', inplace=True)

furnished_mode = df["FURNISHED"].mode()[0]
df["FURNISHED"].fillna(furnished_mode, inplace=True)



#################################
# 7. Feature Extraction
#################################

cat_cols, num_cols, cat_but_car = grab_col_names(df)
num_cols.append('TOTALROOM')
num_cols.append('BATHROOMS')


df[df['NETMETREKARE'] == 1]
df = df.drop(2822)


df['ODA_BASİNA_ORT_METREKARE'] = df['NETMETREKARE'] / df['TOTALROOM']
df['NEW_FLOOR_RATIO'] = df['FLOORNUMBER'] / df['FLOOR']
df["BATHROOM_BEDROOM_RATIO"] = df['BATHROOMS'] / df['TOTALROOM']
df["PRICE_PER_SQR"] = df["PRICE"] / df["NETMETREKARE"]
df['FLOOR_AGE_INTERACTION'] = df['FLOOR'] * df['AGE']


# AGE LEVEL
df.loc[(df["AGE"] <= 10), "AGE_LEVEL"] = "Yeni"
df.loc[(df["AGE"] > 10) & (df['AGE'] <= 30), "AGE_LEVEL"] = "Orta"
df.loc[(df["AGE"] > 30), "AGE_LEVEL"] = "Eski"


cat_cols, num_cols, cat_but_car = grab_col_names(df)


######################################
# RARE
######################################

cat_cols.remove("TOTALROOM")
cat_cols.remove("BATHROOMS")


def rare_analyser(dataframe, target, cat_cols):
    for col in cat_cols:
        print(col, ":", len(dataframe[col].value_counts()))
        print(pd.DataFrame({"COUNT": dataframe[col].value_counts(),
                            "RATIO": dataframe[col].value_counts() / len(dataframe),
                            "TARGET_MEAN": dataframe.groupby(col)[target].mean()}), end="\n\n\n")

rare_analyser(df, "PRICE", cat_cols)



def rare_encoder(dataframe, rare_perc):
    temp_df = dataframe.copy()

    rare_columns = [col for col in temp_df.columns if temp_df[col].dtypes == 'O'
                    and (temp_df[col].value_counts() / len(temp_df) < rare_perc).any(axis=None)]

    for var in rare_columns:
        tmp = temp_df[var].value_counts() / len(temp_df)
        rare_labels = tmp[tmp < rare_perc].index
        temp_df[var] = np.where(temp_df[var].isin(rare_labels), 'Rare', temp_df[var])

    return temp_df


dff = rare_encoder(df, 0.01)
rare_analyser(dff, "PRICE", cat_cols)



##################################################
# Label Encoding & One-Hot Encoding
##################################################

binary_cols = ["FURNISHED"]
ohe_cols = ['BUILDINGTYPE', 'USAGESTATUS', 'FUELTYPE', 'AGE_LEVEL',  'HEATINGSYSTEM']

def one_hot_encoder(dataframe, categorical_cols, drop_first=True):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

dff = one_hot_encoder(dff, ohe_cols)

label_encoder = LabelEncoder()
dff["FURNISHED"] = label_encoder.fit_transform(dff["FURNISHED"])


######################################################
# Multiple Linear Regression
######################################################

X = dff.drop("PRICE", axis=1)
y = dff[["PRICE"]]

##########################
# Model
##########################

# holdout yöntemi
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

reg_model = LinearRegression()
reg_model.fit(X_train, y_train)

# sabit
reg_model.intercept_

# coefficients
reg_model.coef_


##########################
# Tahmin Başarısını Değerlendirme
##########################

# Train RMSE
y_pred = reg_model.predict(X_train)
np.sqrt(mean_squared_error(y_train, y_pred))
#7305


# TRAIN RKARE
reg_model.score(X_train, y_train)
#0.8948


# Test RMSE
y_pred = reg_model.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred))
#7265


# Test RKARE
reg_model.score(X_test, y_test)
#0.84


# 10 Katlı CV RMSE
np.mean(np.sqrt(-cross_val_score(reg_model,
                                 X, y,
                                 cv=10,
                                 scoring="neg_mean_squared_error")))
#7396


# Örnek yeni veri
new_data = X.sample(n=1, random_state=8)

# Tahmin
new_prediction = reg_model.predict(new_data)
print("Yeni veri için tahmin:", new_prediction)
# Yeni veri için tahmin: [[17070.73803143]]







