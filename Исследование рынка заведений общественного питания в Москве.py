#!/usr/bin/env python
# coding: utf-8

# # Исследование рынка заведений общественного питания в Москве
# 
# - Автор: Бусс Татьяна Сергеевна
# - Дата: 05.07.2025

# ### Цели и задачи проекта
# 
# Оснавная цель проекта провести исследовательский анализ рынка заведений общественного питания в городе Москва для выбора направления и расположения нового заведения. 
# 
# Задачи:
# 
# 1. Загрузить данные и познакомиться с их содержимым.
# 2. Провести предобработку данных.
# 3. Провести исследовательский анализ данных.
# 4. Сформулировать выводы по проведённому анализу
# 

# ### Описание данных
# 
# Для анализа имеются данные из двух датасетов с информацией из открытых источников за 2022 год.
# 
# /datasets/rest_info.csv - содержит основную информацию о заведениях.
# /datasets/rest_price.csv - содержит информацию о среднем чеке в заведениях.
# 
# ### Данные датасета /datasets/rest_info.csv: 
# - name — название заведения;
# - address — адрес заведения;
# - district — административный район, в котором находится заведение:
# - category — категория заведения;
# - hours — информация о днях и часах работы;
# - rating — рейтинг заведения по оценкам пользователей в Яндекс Картах (высшая оценка — 5.0);
# - chain — число, выраженное 0 или 1, которое показывает, является ли заведение сетевым:0 — заведение не является сетевым; 1 — заведение является сетевым.
# - seats — количество посадочных мест.
# 
# ### Данные датасета /datasets/rest_price.csv:
# - price — категория цен в заведении;
# - avg_bill — строка, которая хранит среднюю стоимость заказа в виде диапазона;
# - middle_avg_bill — число с оценкой среднего чека;
# - middle_coffee_cup — число с оценкой одной чашки капучино.

# ### Содержание проекта
# 
# 1. Загрузка данных и знакомство с ними.
# 2. Предобработка данных.
# 3. Исследовательский анализ данных.
# 4. Итоговые выводы.
# ---

# ## 1. Загрузка данных и знакомство с ними
# 

# In[369]:


get_ipython().system('pip install phik #устанавливаю библиотеку для расчета коэффециента корреляции')


# In[370]:


#Подключаю необходимые библиотеки
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#Загружаю библиотеку для расчёта коэффициента корреляции phi_k
from phik import phik_matrix


# In[371]:


#Загружаю датасеты
rest_info = pd.read_csv('https://code.s3.yandex.net/datasets/rest_info.csv')
rest_price = pd.read_csv('https://code.s3.yandex.net/datasets/rest_price.csv')


# In[372]:


#Выгружаю 5 первых строк из датафрейма 
rest_info.head()


# In[373]:


#Вывожу информацию о первом датафрейме
rest_info.info()


# In[374]:


#Выгружаю первые 5 строк второго датафрейма
rest_price.head()


# In[375]:


#Выгружаю информацию о втором датафрейме
rest_price.info()


# ---
# 
# ### Промежуточный вывод
# 
# Датафрейм rest_info содержит 9 столбцов и 8406 строк. Названия имеют приемлемый для анализа вид. Все данные хранятся в типах object, float64 и int64. Типы данных считаю подходящими для анализа. Пропуски содержатся в столцах hours и seats, но стоит проверить нет ли в других столбцах значений-индикаторов. Данные соответствуют описанию.
# 
# Датафрейм rest_price содержит 5 столбцов и 4058 строк. Названия столбцов имею приемлемый для анализа вид и приведены к стилю snake_case. Все данные хранятся в типах object и float64 и являются подходящими для анализа. Пропуски соержатся в столбцах middle_coffee_cup, middle_avg_bill, avg_bill и price. Пропусков достаточно большоое количество. Данные соответствуют описанию

# ### Подготовка единого датафрейма
# 
# 

# In[376]:


#Соединяю два датасета в один
df = rest_info.merge(rest_price, on='id', how='left')


# ## 2. Предобработка данных
# 

# In[377]:


df.dtypes


# В типах данных нет критических ошибок. Можно привести столбец seats к целочисленному формату только потому, что количество посадочных мест не может быть дробным числом.

# In[378]:


#Использую errors т.к. пропуски еще не обработаны
df['seats'] = pd.to_numeric(df['seats'], errors='coerce').astype('Int64') 
df.dtypes


# In[379]:


abs_missing = df.isna().sum() #количество абсолютных пропусков
rel_missing = (df.isna().mean()) * 100 #доля пропусков
missing_data = pd.DataFrame({
    'Absolute Missing': abs_missing,
    'Relative Missing (%)': rel_missing}) #создаю датафрейм для удобного отображения
missing_data_sorted = missing_data.sort_values(by='Absolute Missing', ascending=False)
print(missing_data_sorted)


# Так как данные о режиме работы, количестве посадочных мест и среднем чеке брались из открытых источников, эта информация просто могла быть не указана и не найдена. Отсюда и слишком много пропусков. Удалить их нельзя т.к. потеряется много ценной информации. Заменю их на значения индикаторы.

# In[380]:


indicator_value = -5 #Для числовых столбцов
columns_to_fill = ['middle_avg_bill', 'middle_coffee_cup', 'seats']
for column in columns_to_fill:
    if (df[column] == indicator_value).any():
        print(f"В столбце '{column}' есть значения, равные {indicator_value}.") #провожу проверку на наличие значений индикатора в столбце 
    else:
        df[column] = df[column].fillna(indicator_value)


# In[381]:


indicator_value = 'нет данных' #для типа object
columns_to_fill = ['price', 'avg_bill', 'hours']
for column in columns_to_fill:
    if (df[column] == indicator_value).any():
        print(f"В столбце '{column}' есть значения, равные {indicator_value}.") #провожу проверку на наличие значений индикатора в столбце 
    else:
        df[column] = df[column].fillna(indicator_value)


# In[382]:


#проверю что пропуски заменены
df.isna().sum()


# In[383]:


#Проверяю на явные дубликаты
initial_row_count = df.shape[0]
print(f'Количество строк до удаления дубликатов: {initial_row_count}')

duplicates = df[df.duplicated()]
print("Дублирующиеся строки:")
print(duplicates)

df_cleaned = df.drop_duplicates()
final_row_count = df_cleaned.shape[0]
print(f'Количество строк после удаления дубликатов: {final_row_count}')


# В датафрейме нет явных дубликатов  

# In[384]:


df['name'] = df['name'].str.lower()
df['address'] = df['address'].str.lower()


# In[385]:


#Проверяю на неявные дубликаты
unique_name = df['name'].unique()
unique_name_count = df['name'].nunique()
total = len(df)
print(unique_name, unique_name_count, total)

unique_address = df['address'].unique()
unique_address_count = df['address'].nunique()
print(unique_address, unique_address_count, total)

unique_id = df['id'].unique()
unique_id_count = df['id'].nunique()
print(unique_id, unique_id_count, total)

# Проверяем дубли по связке name и address
duplicates = df[df.duplicated(subset=['name', 'address'], keep=False)]

# Количество дублирующих строк
num_duplicates = len(duplicates)
print(duplicates, num_duplicates)
df_cleaned = df.drop_duplicates(subset=['name', 'address'])

df = df_cleaned.reset_index(drop=True)

print(f"Количество записей после удаления дублей: {len(df)}")


# - Для дальнейшей работы создайте столбец `is_24_7` с обозначением того, что заведение работает ежедневно и круглосуточно, то есть 24/7:
#   - логическое значение `True` — если заведение работает ежедневно и круглосуточно;
#   - логическое значение `False` — в противоположном случае.

# In[386]:


#Создаю столбец is_24_7 где 1-круглосуточное, 0-не круглосуточное
def is_24_7(hours_str):
    if 'ежедневно' in hours_str and 'круглосуточно' in hours_str:
        return 1
    return 0
df['is_24_7'] = df['hours'].apply(is_24_7)
df.head(5)


# ---
# 
# ### Промежуточный вывод
# 

# Типы данных были оптимизированы, пропуски заменены на значения-индикаторы, явных дубликатов обнаружено не было

# ## 3. Исследовательский анализ данных
# 

# ---
# 
# ### Задача 1
# 
# Какие категории заведений представлены в данных? Исследуйте количество объектов общественного питания по каждой категории. Результат сопроводите подходящей визуализацией.

# In[387]:


counts = df['category'].value_counts()
perc = df['category'].value_counts(normalize=True) * 100

distribution_df = pd.DataFrame({'Количество': counts,'Процент': perc})
print('Распределение по категориям:')
print(distribution_df)


# In[388]:


plt.figure(figsize=(8, 4))

df['category'].value_counts().plot(
               kind='bar',
               rot=45, 
               legend=False, 
               title=f'Распределение заведений по категориям')

plt.xlabel('Категории')
plt.ylabel('Количество заведений')
plt.grid(axis = 'y')
plt.show()


# In[389]:


plt.figure(figsize=(8, 4))
category_counts = df['category'].value_counts(normalize=True) * 100
category_counts.plot(
    kind='bar',
    rot=45,
    legend=False,
    title='Распределение заведений по категориям')

plt.xlabel('Категории')
plt.ylabel('Процент заведений')
plt.grid(axis='y')
plt.show()


# Самое большое количество заведений в категориях кафе, ресторан и кофейня.

# ---
# 
# ### Задача 2
# 
# Какие административные районы Москвы присутствуют в данных? Исследуйте распределение количества заведений по административным районам Москвы, а также отдельно распределение заведений каждой категории в Центральном административном округе Москвы. Результат сопроводите подходящими визуализациями.

# In[390]:


print('Распределение объектов по районам Москвы')
df['district'].value_counts()


# In[391]:


plt.figure(figsize=(8, 4))

df['district'].value_counts().plot(
               kind='bar',
               rot=90, 
               legend=False, 
               title=f'Распределение заведений по районам')

plt.xlabel('Районы')
plt.ylabel('Количество заведений')
plt.grid(axis='y')
plt.show()


# In[392]:


plt.figure(figsize=(8, 4))
category_counts = df['district'].value_counts(normalize=True) * 100
category_counts.plot(
    kind='bar',
    rot=90,
    legend=False,
    title='Распределение заведений по районам')

plt.xlabel('Районы')
plt.ylabel('Процент заведений')
plt.grid(axis='y')
plt.show()


# In[393]:


central_df = df[df['district'] == 'Центральный административный округ']
category_counts_central = central_df['category'].value_counts()

plt.figure(figsize=(8,5))
category_counts_central.plot(kind='bar')
plt.title('Распределение заведений по категориям в Центральном административном округе Москвы')
plt.xlabel('Категория')
plt.ylabel('Количество заведений')
plt.grid(axis='y')
plt.show()


# In[394]:


central_df = df[df['district'] == 'Центральный административный округ']
category_counts_central = central_df['category'].value_counts(normalize=True) * 100

plt.figure(figsize=(8,5))
category_counts_central.plot(kind='bar')
plt.title('Распределение заведений по категориям в Центральном административном округе Москвы')
plt.xlabel('Категория')
plt.ylabel('Количество заведений')
plt.grid(axis='y')
plt.show()


# В центральном районе самое большое количество заведений в категориях ресторан, кафе и кофейня.

# ---
# 
# ### Задача 3
# 
# Изучите соотношение сетевых и несетевых заведений в целом по всем данным и в разрезе категорий заведения. Каких заведений больше — сетевых или несетевых? Какие категории заведений чаще являются сетевыми? Исследуйте данные, ответьте на вопросы и постройте необходимые визуализации.

# In[395]:


total_counts = df['chain'].value_counts()

print("Общее количество заведений:")
print(total_counts)


# In[396]:


total_counts.plot.pie(autopct='%1.1f%%', title='Общее соотношение сетевых и несетевых заведений')
plt.ylabel('')
plt.show()


# In[397]:


category_counts = df.groupby(['chain', 'category']).size().unstack()

print("Количество заведений по категориям и типам:")
print(category_counts)


# In[398]:


category_counts.plot(kind='bar', stacked=True, figsize=(9,5),rot=0)
plt.title('Соотношение сетевых и несетевых заведений по категориям')
plt.xlabel('Категория')
plt.ylabel('Количество')
plt.show()


# In[399]:


category_diffs = category_counts.div(category_counts.sum(axis=1), axis=0) * 100
ax = category_diffs.plot(
    kind='bar',
    stacked=True,
    figsize=(8,6),
    rot=90)

plt.title('Доля сетевых и несетевых заведений по категориям')
plt.xlabel('Категория')
plt.ylabel('Доля (%)')
plt.legend(title='Тип сети')
plt.grid(axis='y')
plt.tight_layout()
plt.show()


# Несетевых заведений больше, чем сетевых. Больше всего сетевых заведений в категориях кафе, ресторан, кофейня. 

# ---
# 
# ### Задача 4
# 
# Исследуйте количество посадочных мест в заведениях. Встречаются ли в данных аномальные значения или выбросы? Если да, то с чем они могут быть связаны? Приведите для каждой категории заведений наиболее типичное для него количество посадочных мест. Результат сопроводите подходящими визуализациями.
# 

# In[400]:


df['seats'].describe() #значение индикатор = -5 влияет на распределение. отфильтрую без него. 


# In[401]:


filt_df = df[df['seats'] != -5]


# In[402]:


filt_df['seats'].describe()


# In[403]:


filt_df['seats'].plot(
                kind='hist',
                bins=25, 
                alpha=0.75,
                edgecolor='black',
                rot=0,)
plt.title('Распределение количества посадочных мест (без значений -5)')
plt.xlabel('Количество посадочных мест')
plt.ylabel('Частота')
plt.show()


# In[404]:


Q1 = filt_df['seats'].quantile(0.25)
Q3 = filt_df['seats'].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = filt_df[(filt_df['seats'] < lower_bound) | (filt_df['seats'] > upper_bound)]
print(f"Обнаружено {len(outliers)} выбросов.")


# In[405]:


mode_per_category = filt_df.groupby('category')['seats'].agg(lambda x: x.mode().iloc[0])
median_per_category = filt_df.groupby('category')['seats'].median()
print("Наиболее типичное количество посадочных мест по категориям (мода):")
print(mode_per_category)
print("\nМедианное количество посадочных мест по категориям:")
print(median_per_category)


# In[406]:


# Визуализация по категориям 
plt.figure(figsize=(12,8))
sns.boxplot(x='category', y='seats', data=filt_df)
plt.title('Распределение посадочных мест по категориям (без значений -5)')
plt.xlabel('Категория')
plt.ylabel('Количество посадочных мест')
plt.xticks(rotation=45)
plt.show()


# В основном данные распределены в промежутке от 0 до 300 и большое количество хвостов в верхней части графика. Это значит что в данных есть выбросы. Теоретически заведения с большим количеством существуют и могли попасть в выборку, но их не большое количество. Поэтому данные от выбросов не очищены.

# ---
# 
# ### Задача 5
# 
# Исследуйте рейтинг заведений. Визуализируйте распределение средних рейтингов по категориям заведений. Сильно ли различаются усреднённые рейтинги для разных типов общепита?

# In[407]:


mean_ratings = df.groupby('category')['rating'].mean().reset_index()

plt.figure(figsize=(10, 6))
sns.barplot(x='rating', y='category', data=mean_ratings.sort_values('rating', ascending=False))
plt.xlabel('Средний рейтинг')
plt.ylabel('Категория')
plt.title('Распределение средних рейтингов по категориям заведений')
plt.show()


# Средний рейтинг заведений находится выше 4 баллов и отличия по рейтингу не значительные.

# ---
# 
# ### Задача 6
# 
# Изучите, с какими данными показывают самую сильную корреляцию рейтинги заведений? Постройте и визуализируйте матрицу корреляции рейтинга заведения с разными данными: его категория, положение (административный район Москвы), статус сетевого заведения, количество мест, ценовая категория и признак, является ли заведения круглосуточным. Выберите самую сильную связь и проверьте её.

# In[408]:


corr_matrix = df[['category', 'district', 'chain', 'seats', 'price',
                         'is_24_7', 'rating']].phik_matrix()

# Выводим результат
print('Корреляционная матрица с коэффициентом phi_k для переменной rating')
corr_matrix.loc[corr_matrix.index != 'rating'][['rating']].sort_values(by='rating', ascending=False)


# Наибольшая корреляция с ценовой категорией (0,29), меньше с районом (0,2) и категорией (0,19). 

# In[409]:


plt.figure(figsize=(2, 6))
data_heatmap = corr_matrix.loc[corr_matrix.index != 'rating'][['rating']].sort_values(by='rating', ascending=False)
sns.heatmap(data_heatmap,
            annot=True,
            fmt='.2f',
            cmap='coolwarm', 
            linewidths=0.5, 
            cbar=False)

plt.title('Тепловая карта коэффициента phi_k \n для данных рейтинга')
plt.xlabel('Рейтинг')

# Выводим график
plt.show()


# In[410]:


plt.figure(figsize=(7, 3))
df.groupby('price')['rating'].mean().sort_values(ascending=False).plot(
               kind='bar',
               rot=90, 
               legend=False,
               title=f'Средний рейтинг в зависимости от ценовой категории')
plt.ylabel('Средний рейтинг')
plt.grid()
plt.show()


# In[411]:


plt.figure(figsize=(7, 3))
df.groupby('district')['rating'].mean().sort_values(ascending=False).plot(
               kind='bar',
               rot=90, 
               legend=False,
               title=f'Средний рейтинг в зависимости от района')
plt.ylabel('Средний рейтинг')
plt.grid()
plt.show()


# Исходя из графиков видно, что действительно чем выше ценовая категория, тем выше рейтинг заведения. А также видно, что в ЦАО действительно рейтинг выше.

# ---
# 
# ### Задача 7
# 
# Сгруппируйте данные по названиям заведений и найдите топ-15 популярных сетей в Москве. Для них посчитайте значения среднего рейтинга. Под популярностью понимается количество заведений этой сети в регионе. К какой категории заведений они относятся? Результат сопроводите подходящими визуализациями.

# In[412]:


df_network = df[df['chain'] == 1]
name_count = df_network.groupby('name').size().reset_index(name='count')
top_15 = name_count.sort_values(by='count', ascending=False).head(15)
print(top_15)


# In[413]:


top_15_with_rating = df_network[df_network['name'].isin(top_15['name'])].groupby('name')['rating'].mean().reset_index()
print(top_15_with_rating)


# In[414]:


categories = df_network[df_network['name'].isin(top_15['name'])].groupby(['name', 'category']).size().reset_index(name='counts')
categories_sorted = categories.sort_values(['name', 'counts'], ascending=[True, False])
categories_top = categories_sorted.groupby('name').first().reset_index()


# In[415]:


result = top_15.merge(top_15_with_rating, on='name')
result = result.merge(categories_top[['name', 'category']], on='name')


# In[416]:


plt.figure(figsize=(10,6))
sns.barplot(x='count', y='name', data=top_15)
plt.title('Топ-15 популярных сетей в Москве по количеству заведений')
plt.xlabel('Количество заведений')
plt.ylabel('Название сети')
plt.show()


# In[417]:


plt.figure(figsize=(12,6))
sns.barplot(x='rating', y='name', data=result)
plt.title('Средний рейтинг топ-15 сетей в Москве')
plt.xlabel('Средний рейтинг')
plt.ylabel('Название сети')
plt.show()


# In[418]:


display(result)


# Сформирован рейтинг топ-15 по сетевым заведением, а также определена их категория. В лидерах по количеству кофейни и пиццерии.

# ---
# 
# ### Задача 8
# 
# Изучите вариацию среднего чека заведения (столбец `middle_avg_bill`) в зависимости от района Москвы. Проанализируйте цены в Центральном административном округе и других. Как удалённость от центра влияет на цены в заведениях? Результат сопроводите подходящими визуализациями.
# 

# In[419]:


df = df[df['middle_avg_bill'] != -5].copy()
df['is_central'] = df['district'].isin(['Центральный административный округ'])
central = df[df['is_central']]
others = df[~df['is_central']]

central_mean = central['middle_avg_bill'].mean()
others_mean = others['middle_avg_bill'].mean()

print(f"Средний чек в ЦАО: {central_mean:.2f}")
print(f"Средний чек в других районах: {others_mean:.2f}")

district_stats = df.groupby('district')['middle_avg_bill'].mean().sort_values()


# In[420]:


plt.figure(figsize=(10, 6))
sns.boxplot(x='is_central', y='middle_avg_bill', data=df)
plt.xticks([0, 1], ['Другие районы', 'Центральный округ'])
plt.title('Распределение среднего чека по районам Москвы')
plt.ylabel('Средний чек')
plt.xlabel('Район')
plt.show()


# In[421]:


df_clean = df[df['middle_avg_bill'] <=3500]

plt.figure(figsize=(10, 6))
sns.boxplot(x='is_central', y='middle_avg_bill', data=df_clean)
plt.xticks([0, 1], ['Другие районы', 'Центральный округ'])
plt.title('Распределение среднего чека по районам Москвы (без значений больше 3500)')
plt.ylabel('Средний чек')
plt.xlabel('Район')
plt.show()


# In[422]:


top_districts = district_stats.sort_values(ascending=False)

plt.figure(figsize=(12, 6))
sns.barplot(x=top_districts.values, y=top_districts.index)
plt.title('Топ-10 районов по среднему чеку')
plt.xlabel('Средний чек')
plt.ylabel('Район')
plt.show()


# По имеющимся данным видно, что средний чек выше в ЦАО и ЗАО, чем в остальных районах.

# ---
# 
# ### Промежуточный вывод
# 

# Топ 3 заведений по категориям: кафе, ресторан и кофейня. Самое большое количество заведений в Центральном районе Москвы. В среднем в заведениях до 100 посадочных мест. Рейтинг и средний чек показывают сильную зависимость от района города: в ЦАО средний чек значительно выше, чем в других районах. Несетевых заведений больше, чем сетевых. Самое большое количество сетевых заведений в категорях кофейня и пиццерия.

# ## 4. Итоговый вывод и рекомендации
# 

# Перед началом анализа проведена предабработка данных: датафреймы объеденены в один, типы данных проверены на соответствие и оптимизированы, обработаны пропуски в данных. 
# 
# Также проведен исследовательский анализ данных и сформулирован вывод. Исходя из представленных данных конкурировать на рынке в ЦАО в категориях кафе и ресторан будет сложнее, поэтому стоит рассмотреть другие районы или другую категорию. Можно рассмотреть для открытия такую категорию как бар, паб в ЦАО. 
