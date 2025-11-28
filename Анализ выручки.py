#!/usr/bin/env python
# coding: utf-8

# Автор: Бусс Татьяна Сергеевна
# Дата. 19.10.2025
# 

# # Анализ выручки и проверка гипотез Яндекс.Афиши

# ### Содержание проекта

# 1. Описание данных
# 2. Загрузка и знакомство с данными
# 3. Предобработка данных и подготовка их к исследованию
# 4. Исследовательский анализ данных
# 5. Статистический анализ данных
# 6. Общий вывод и рекомендации

# ### Описание данных
# 

# Первый датасет final_tickets_orders_df.csv включает информацию обо всех заказах билетов, совершённых с двух типов устройств — мобильных и стационарных.
# 
# - order_id — уникальный идентификатор заказа.
# - user_id — уникальный идентификатор пользователя.
# - created_dt_msk — дата создания заказа (московское время).
# - created_ts_msk — дата и время создания заказа (московское время).
# - event_id — идентификатор мероприятия из таблицы events.
# - cinema_circuit — сеть кинотеатров. Если не применимо, то здесь будет значение 'нет'.
# - age_limit — возрастное ограничение мероприятия.
# - currency_code — валюта оплаты, например rub для российских рублей.
# - device_type_canonical — тип устройства, с которого был оформлен заказ, например mobile для мобильных устройств, desktop для стационарных.
# - revenue — выручка от заказа.
# - service_name — название билетного оператора.
# - tickets_count — количество купленных билетов.
# - total — общая сумма заказа.

# Второй датасет final_tickets_events_df содержит информацию о событиях, включая город и регион события, а также информацию о площадке проведения мероприятия. 
# - event_id — уникальный идентификатор мероприятия.
# - event_name — название мероприятия. Аналог поля event_name_code из исходной базы данных.
# - event_type_description — описание типа мероприятия.
# - event_type_main — основной тип мероприятия: театральная постановка, концерт и так далее.
# - organizers — организаторы мероприятия.
# - region_name — название региона.
# - city_name — название города.
# - venue_id — уникальный идентификатор площадки.
# - venue_name — название площадки.
# - venue_address — адрес площадки.

# ### Загрузка и знакомство с данными

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
from scipy.stats import ttest_rel
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import proportion_effectsize
from math import ceil
from statsmodels.stats.proportion import proportions_ztest
import seaborn as sns
from scipy import stats
get_ipython().system('pip install missingno')
import missingno as msno
from io import BytesIO, StringIO
import zipfile
import requests
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import proportion_effectsize
import statsmodels.api as sm
import numpy as np


# In[2]:


o_df = pd.read_csv('https://code.s3.yandex.net//datasets/final_tickets_orders_df.csv')
e_df = pd.read_csv('https://code.s3.yandex.net//datasets/final_tickets_events_df.csv')
rate_df = pd.read_csv('https://code.s3.yandex.net//datasets/final_tickets_tenge_df.csv')
display(o_df.head())
display(e_df.head())
display(rate_df.head())


# In[3]:


display(o_df.info())
display(e_df.info())
display(rate_df.info())


# In[4]:


o_df['currency_code'].nunique()


# В данных почти нет пропусков, кроме столбца days_since_prev. Данные полные и достаточные для дальнейшего анализа. Соответвствуют описанию

# In[5]:


print(o_df.dtypes, '\n', e_df.dtypes)


# In[6]:


o_df['created_dt_msk'] = pd.to_datetime(o_df['created_dt_msk'])
o_df['created_ts_msk'] = pd.to_datetime(o_df['created_ts_msk'])


# При проверке типов данных обнаружено, что столбцы с датой имеют тип объект. Было принято решение привести их к типу дата, для удобного дальнейшего анализа.

# In[7]:


msno.matrix(o_df)
plt.title('Матрица пропусков')
plt.show()
missing_stats = pd.DataFrame({'Кол-во пропусков': o_df.isnull().sum(),
    'Доля пропусков': o_df.isnull().sum() / len(o_df)}).sort_values(by='Доля пропусков', ascending=False)
display(missing_stats.style.background_gradient(cmap='coolwarm'))


# In[8]:


msno.matrix(e_df)
plt.title('Матрица пропусков')
plt.show()
missing_stats = pd.DataFrame({'Кол-во пропусков': e_df.isnull().sum(),
    'Доля пропусков': e_df.isnull().sum() / len(e_df)}).sort_values(by='Доля пропусков', ascending=False)
display(missing_stats.style.background_gradient(cmap='coolwarm'))


# In[9]:


categorical_cols = ['cinema_circuit', 'device_type_canonical', 'currency_code', 'service_name']

for col in categorical_cols:
    print(f"{col} уникальные значения: {o_df[col].unique()}")
    o_df[col] = o_df[col].fillna('неизвестно')
    o_df[col] = o_df[col].astype('category')


# In[10]:


o_df['revenue'].hist(bins=50)
plt.title('Распределение revenue')
plt.show()

o_df['tickets_count'].hist(bins=20)
plt.title('Распределение tickets_count')
plt.show()
q99 = o_df['revenue'].quantile(0.99)
o_df_clean = o_df[o_df['revenue'] <= q99]


# In[11]:


o_df['days_since_prev'].hist(bins=50)
plt.title('Распределение days_since_prev')
plt.show()
o_df['days_since_prev'] = o_df['days_since_prev'].fillna(o_df['days_since_prev'].median())


# In[12]:


for col in ['event_type_description', 'event_type_main', 'region_name', 'city_name', 'venue_name', 'organizers']:
    print(f"{col} уникальные значения: {e_df[col].unique()}")
    e_df[col] = e_df[col].fillna('неизвестно')
    e_df[col] = e_df[col].astype('category')


# Вывод: Категориальные данные обладают высокой кардинальностью (много уникальных значений). В названиях мероприятий и организаторов присутствуют структурированные элементы (например, номера, слова с определенным значением), что позволяет извлечь дополнительные признаки.

# In[13]:


rub_df = o_df[o_df['currency_code'] == 'rub']
kzt_df = o_df[o_df['currency_code'] == 'kzt']
rub_df['revenue'].describe()


# In[14]:


kzt_df['revenue'].describe()


# Вывод: У цен в рублях минимум -90,76. Вероятно был возврат за стоимость билетов. Среднее и медианные значения разнятся. В тенге минимум 0. Вероятно билет подарочный. 

# In[15]:


duplicates = o_df[o_df.duplicated()]
print(f"Количество явных дубликатов: {len(duplicates)}")


# In[16]:


duplicates = e_df[e_df.duplicated()]
print(f"Количество явных дубликатов: {len(duplicates)}")


# In[17]:


all_cols = o_df.columns.tolist()
key_cols = ['user_id', 'created_ts_msk', 'event_id', 'tickets_count', 'total', 'device_type_canonical']
check_cols = [col for col in all_cols if col != 'order_id']
duplicates_mask = o_df.duplicated(subset=check_cols, keep=False)
potential_duplicates = o_df[duplicates_mask]
print("Потенциальные дубликаты:")
print(potential_duplicates.head())
orders_df_sorted = o_df.sort_values(by=key_cols + ['created_ts_msk'])
orders_df_clean = orders_df_sorted.drop_duplicates(subset=key_cols, keep='last')


# In[18]:


original_size = len(o_df)
clean_size = len(orders_df_clean)
removed_count = original_size - clean_size
removed_percentage = (removed_count / original_size) * 100


print(f"\nСтатистика очистки данных:")
print(f"Исходных строк: {original_size}")
print(f"Уникальных строк: {clean_size}")
print(f"Удалено строк: {removed_count} ({removed_percentage:.2f}%)")


print(f"\nУникальных event_id: {o_df['event_id'].nunique()}, всего строк: {len(o_df)}")
duplicates_event_id = o_df[o_df.duplicated(subset=['event_id'], keep=False)]


# In[19]:


e_df['event_name_norm'] = e_df['event_name'].str.lower().str.replace(r'\s+', '', regex=True)
duplicates_mask = e_df.duplicated(subset=['event_name_norm', 'venue_id'], keep=False)
potential_duplicates = e_df[duplicates_mask].sort_values(by=['event_name_norm', 'venue_id'])

display(potential_duplicates)


# In[20]:


e_df['event_name_norm'] = e_df['event_name'].str.lower().str.replace(r'\s+', '', regex=True)
original_size = len(e_df)
duplicates_mask = e_df.duplicated(subset=['event_name_norm', 'venue_id'], keep=False)
potential_duplicates = e_df[duplicates_mask].sort_values(by=['event_name_norm', 'venue_id'])
e_df_clean = e_df.drop_duplicates(subset=['event_name_norm', 'venue_id'], keep='first')
clean_size = len(e_df_clean)
removed_count = original_size - clean_size
removed_percentage = (removed_count / original_size) * 100


# In[21]:


print("\nСтатистика очистки данных:")
print(f"Исходных записей: {original_size}")
print(f"Уникальных записей: {clean_size}")
print(f"Удалено дубликатов: {removed_count}")
print(f"Доля удаленных данных: {removed_percentage:.2f}%")

print("\nПотенциальные дубликаты:")
print(potential_duplicates)

remaining_duplicates_mask = e_df_clean.duplicated(
    subset=['event_name_norm', 'venue_id'],
    keep=False
)
print(f"\nОсталось дубликатов: {remaining_duplicates_mask.sum()}")

e_df = e_df_clean.copy()


# Вывод: Обработаны дубликаты явные и неявные. Данные чисты и готовы к объединению. 

# In[22]:


e_df['event_name_norm'] = e_df['event_name'].str.lower().str.replace(r'\s+', '', regex=True)


# In[23]:


df = pd.merge(orders_df_clean, e_df_clean, on='event_id', how='left') 


# In[24]:


plt.figure(figsize=(8, 4))
sns.boxplot(x=rub_df['revenue'])
plt.title('Распределение выручки в рублях')
plt.xlabel('общая выручка в рублях')
plt.show()


# In[25]:


rub_q99 = rub_df['revenue'].quantile(0.99)

df_by_q99 = rub_df[rub_df['revenue'] <= q99]
plt.figure(figsize=(8, 4))
sns.boxplot(x=df_by_q99['revenue'])
plt.title('Распределение выручки в рублях (после фильтрации)')
plt.xlabel('общая выручка в рублях')
plt.show()


# In[26]:


plt.figure(figsize=(8, 4))
sns.boxplot(x=kzt_df['revenue'])
plt.title('Распределение выручки в тенге')
plt.xlabel('общая выручка в тенге')
plt.show()


# In[27]:


kzt_q99 = kzt_df['revenue'].quantile(0.99)

df_by_q99 = kzt_df[kzt_df['revenue'] <= q99]
plt.figure(figsize=(8, 4))
sns.boxplot(x=df_by_q99['revenue'])
plt.title('Распределение выручки в рублях (после фильтрации)')
plt.xlabel('общая выручка в рублях')
plt.show()


# In[28]:


df_rub_filtered = rub_df[rub_df['revenue'] <= rub_q99].copy()
df_kzt_filtered = kzt_df[kzt_df['revenue'] <= kzt_q99].copy()
df_rub_filtered['currency_code'] = 'rub'
df_kzt_filtered['currency_code'] = 'kzt'
combined_df = pd.concat([df_rub_filtered, df_kzt_filtered], ignore_index=True)


# In[29]:


combined_df.head()


# In[30]:


combined_df['data'] = pd.to_datetime(rate_df['data'])
df = pd.merge(combined_df, rate_df, how='left', left_on='currency_code', right_on = 'cdx' )
mdf = pd.merge(df,e_df, how='left', on='event_id')
mdf.head()


# In[31]:


mdf['created_dt_msk'] = pd.to_datetime(df['created_dt_msk'])
mdf['month'] = df['created_dt_msk'].dt.month


# In[32]:


mdf['revenue_rub'] = np.where(
    mdf['currency_code'] == 'rub',
    mdf['revenue'],
    np.where(
        mdf['currency_code'] == 'kzt',
        mdf['revenue'] * (mdf['curs'] / 100),
        np.nan))


# In[33]:


mdf['one_ticket_revenue_rub'] = mdf['revenue_rub'] / mdf['tickets_count']


# In[34]:


season_dict = {
    1: 'зима',  2: 'зима',  3: 'весна',
    4: 'весна', 5: 'весна', 6: 'лето',
    7: 'лето',  8: 'лето',  9: 'осень',
   10: 'осень', 11: 'осень', 12: 'зима'}

mdf['season'] = mdf['created_dt_msk'].dt.month.map(season_dict)


# In[35]:


mdf.head()


# ВЫВОД: В этой части были загружены и оценены данные, обработаны пропуски. Также типы данных приведены к нормальным.  Обработаны явные и неявные дубликаты. В датафрейме o_df неявных дубликатов 60, что составляет меньше 1 процента от общего числа строк - они удалены. В датафрейме e_df оставлены только уникальные строки. Добавлены необходимые столбцы, выручка приведена к единой валюте.  Данные очищены от выбросов.

# ### Исследовательский анализ данных

# In[36]:


jun_to_nov = mdf[(mdf['month'] >= 6) & (mdf['month'] < 12)]

monthly_orders = jun_to_nov.groupby('month').agg(
    total_orders=('order_id', 'nunique')
).reset_index()

month_map = {
    6: 'Июнь', 7: 'Июль', 8: 'Август', 
    9: 'Сентябрь', 10: 'Октябрь', 11: 'Ноябрь'
}

monthly_orders['month'] = monthly_orders['month'].map(month_map)


# In[37]:


plt.figure(figsize=(12, 6))
sns.lineplot(
    data=monthly_orders,
    x='month',
    y='total_orders',
    marker='o'
)

plt.title('Динамика количества заказов с июня по ноябрь 2024')
plt.xlabel('Месяц')
plt.ylabel('Количество заказов')
plt.grid(True)
plt.tight_layout()
plt.show()

# Выводим статистику
print("Статистика заказов по месяцам:")
print(monthly_orders[['month', 'total_orders']].sort_values('month'))


# Вывод: Самое большое количество заказов было в октябре. Видна четкая положительная динамика от июня к октябрю. В ноябре заказов не было

# In[38]:



# Фильтрация данных по сезонам
summer_df = mdf[mdf['season'].isin(['лето'])]
autumn_df = mdf[mdf['season'].isin(['осень'])]

# Анализ типа мероприятий
summer_event_type = summer_df['event_type_main'].value_counts(normalize=True)
autumn_event_type = autumn_df['event_type_main'].value_counts(normalize=True)

# Анализ устройств
summer_device = summer_df['device_type_canonical'].value_counts(normalize=True)
autumn_device = autumn_df['device_type_canonical'].value_counts(normalize=True)

# Анализ возрастных ограничений
summer_age = summer_df['age_limit'].value_counts(normalize=True)
autumn_age = autumn_df['age_limit'].value_counts(normalize=True)


# In[39]:


season_colors = {'Summer': '#FF6347', 'Autumn': '#4682B4'}  
season_translation = {'Лето': 'Summer', 'Осень': 'Autumn'}
plt.figure(figsize=(16, 12))

# График 1: Тип мероприятия
plt.subplot(2, 2, 1)
summer_event_type.plot(
    kind='bar', 
    color=season_colors[season_translation['Лето']],
    alpha=0.7, 
    label='Лето', 
    position=0, 
    width=0.4)
autumn_event_type.plot(
    kind='bar', 
    color=season_colors[season_translation['Осень']],  
    alpha=0.7, 
    label='Осень', 
    position=1, 
    width=0.4)
plt.title('Распределение по типу мероприятия')
plt.ylabel('Доля заказов')
plt.legend()


# In[40]:


# График 2: Тип устройства
plt.figure(figsize=(16, 12))
plt.subplot(2, 2, 2)
summer_device.plot(
    kind='bar', 
    color=season_colors[season_translation['Лето']],
    alpha=0.7, 
    label='Лето', 
    position=0, 
    width=0.4)
autumn_device.plot(
    kind='bar', 
    color=season_colors[season_translation['Осень']],
    alpha=0.7, 
    label='Осень', 
    position=1, 
    width=0.4)
plt.title('Распределение по типу устройства')
plt.ylabel('Доля заказов')


# In[41]:



# График 3: Возрастной рейтинг
plt.figure(figsize=(10, 8))
plt.subplot(2, 2, 3)
summer_age.plot(
    kind='bar', 
    color=season_colors[season_translation['Лето']],
    alpha=0.7, 
    label='Лето', 
    position=0, 
    width=0.4)
autumn_age.plot(
    kind='bar', 
    color=season_colors[season_translation['Осень']],
    alpha=0.7, 
    label='Осень', 
    position=1, 
    width=0.4)
plt.title('Распределение по возрастному рейтингу')
plt.ylabel('Доля заказов')

plt.tight_layout()
plt.show()


# Вывод: По типу мероприятия доля заказов самая высокая в категории другое и летом и осенью (летом выше), а в категории концерты осенью доля заказов выше. 
#     Больше доля заказов с мобильного приложения и летом и осенью. Чаще покупают мероприятия возрастных групп 6 и 16лет. 

# In[42]:


summer_stats = summer_df.groupby('event_type_main').agg(
    avg_price_summer=('one_ticket_revenue_rub', 'mean')
).reset_index()

autumn_stats = autumn_df.groupby('event_type_main').agg(
    avg_price_autumn=('one_ticket_revenue_rub', 'mean')
).reset_index()


# In[43]:


price_comparison = pd.merge(
    summer_stats, 
    autumn_stats, 
    on='event_type_main', 
    how='outer'
)


# In[44]:


# Рассчитываем процентное изменение
price_comparison['price_change'] = (
    (price_comparison['avg_price_autumn'] - price_comparison['avg_price_summer']) / 
    price_comparison['avg_price_summer'] * 100
)

# Создаем melt с правильным указанием сезонов
melted_data = pd.melt(
    price_comparison,
    id_vars='event_type_main',
    value_vars=['avg_price_summer', 'avg_price_autumn'],
    var_name='season',
    value_name='value'
)


# In[45]:


# Обновляем значения сезона для наглядности
melted_data['season'] = melted_data['season'].map({
    'avg_price_summer': 'Лето',
    'avg_price_autumn': 'Осень'
})


# In[46]:


# График 1: Сравнение средних цен
plt.figure(figsize=(12, 6))
sns.barplot(
    x='event_type_main', 
    y='value', 
    hue='season', 
    data=melted_data,
    palette={'Лето': '#FF6347', 'Осень': '#4682B4'}
)
plt.title('Сравнение средней стоимости билета по сезонам')
plt.xlabel('Тип мероприятия')
plt.ylabel('Средняя цена билета (руб.)')
plt.xticks(rotation=45)
plt.legend(title='Сезон')
plt.tight_layout()


# In[47]:


# График 2: Процентное изменение
plt.figure(figsize=(12, 6))
sns.barplot(
    x='event_type_main', 
    y='price_change', 
    data=price_comparison,
    palette='coolwarm'
)
plt.axhline(0, color='black', linestyle='--')
plt.title('Процентное изменение цены билета осень/лето')
plt.xlabel('Тип мероприятия')
plt.ylabel('Изменение цены (%)')
plt.xticks(rotation=45)
plt.tight_layout()


# In[48]:


# Анализ результатов
print("Статистика по средней цене билета:")
print(price_comparison[['event_type_main', 'avg_price_summer', 'avg_price_autumn', 'price_change']])

# Находим максимальное изменение
max_change = price_comparison['price_change'].max()
min_change = price_comparison['price_change'].min()
print(f"\nМаксимальное увеличение: {max_change:.2f}%")
print(f"Максимальное снижение: {min_change:.2f}%")


# ВЫВОД: Осенью пользователи более активны. Самое большое количество заказов за октябрь. Осенью Стоимость билетов ниже, с мобильных устройств также больше покупок осенью. При этом по возрастным категориям чаще детские мероприятия проходят летом. Осенью скидка на билеты выше, чем летом.

# In[49]:


fall_2024_df = mdf[
    (mdf['created_dt_msk'].dt.year == 2024) &
    (mdf['season'] == 'осень')]

daily_metrics = fall_2024_df.groupby(fall_2024_df['created_dt_msk'].dt.date).agg(
    total_orders=('order_id', 'nunique'), 
    unique_users=('user_id', 'nunique'),   
    total_revenue=('revenue_rub', 'sum'),  
    total_tickets=('tickets_count', 'sum') 
).reset_index()


daily_metrics['avg_orders_per_user'] = daily_metrics['total_orders'] / daily_metrics['unique_users']
daily_metrics['avg_ticket_price'] = daily_metrics['total_revenue'] / daily_metrics['total_tickets']


# In[50]:


plt.style.use('seaborn-whitegrid')
plt.figure(figsize=(16, 8))
plt.subplot(2, 2, 1)
plt.plot(daily_metrics['created_dt_msk'], daily_metrics['total_orders'], label='Заказы', color='blue')
plt.plot(daily_metrics['created_dt_msk'], daily_metrics['unique_users'], label='DAU', color='orange')
plt.title('Динамика заказов и активных пользователей')
plt.xlabel('Дата')
plt.ylabel('Количество')
plt.legend()
plt.grid(True)


# In[51]:


# График 2: Среднее число заказов на пользователя
plt.figure(figsize=(16, 8))
plt.subplot(2, 2, 2)
plt.plot(daily_metrics['created_dt_msk'], daily_metrics['avg_orders_per_user'], color='green')
plt.title('Среднее число заказов на пользователя')
plt.xlabel('Дата')
plt.ylabel('Среднее значение')
plt.grid(True)


# In[52]:


plt.figure(figsize=(16, 8))
plt.subplot(2, 2, 3)
plt.plot(daily_metrics['created_dt_msk'], daily_metrics['avg_ticket_price'], color='purple')
plt.title('Средняя стоимость билета')
plt.xlabel('Дата')
plt.ylabel('Руб.')
plt.grid(True)

plt.tight_layout()
plt.show()


# In[53]:



print("Статистика по заказам:")
print(daily_metrics['total_orders'].describe())

print("\nСтатистика по DAU:")
print(daily_metrics['unique_users'].describe())

print("\nСтатистика по среднему числу заказов на пользователя:")
print(daily_metrics['avg_orders_per_user'].describe())

print("\nСтатистика по средней стоимости билета:")
print(daily_metrics['avg_ticket_price'].describe())


# Вывод: Динамика заказов скачкообразная, но с тенденцией на увеличение. Среднее количество заказов на одного пользователя также скачкообразная, но с тенденцией на увеличение

# In[54]:


fall_2024_df['day_of_week'] = fall_2024_df['created_dt_msk'].dt.dayofweek
fall_2024_df['day_name'] = fall_2024_df['created_dt_msk'].dt.day_name()
fall_2024_df['is_weekend'] = fall_2024_df['day_of_week'].apply(lambda x: 'Weekend' if x >= 5 else 'Weekday')
weekly_metrics = fall_2024_df.groupby('day_of_week').agg(
total_orders=('order_id', 'nunique'),
unique_users=('user_id', 'nunique'),
total_revenue=('revenue_rub', 'sum'),
total_tickets=('tickets_count', 'sum')
).reset_index()

weekly_metrics['avg_orders_per_user'] = weekly_metrics['total_orders'] / weekly_metrics['unique_users']
weekly_metrics['avg_ticket_price'] = weekly_metrics['total_revenue'] / weekly_metrics['total_tickets']


# In[55]:



weekend_metrics = fall_2024_df.groupby('is_weekend').agg(
total_orders=('order_id', 'nunique'),
unique_users=('user_id', 'nunique'),
total_revenue=('revenue_rub', 'sum'),
total_tickets=('tickets_count', 'sum')
).reset_index()

weekend_metrics['avg_orders_per_user'] = weekend_metrics['total_orders'] / weekend_metrics['unique_users']
weekend_metrics['avg_ticket_price'] = weekend_metrics['total_revenue'] / weekend_metrics['total_tickets']


# In[56]:


plt.figure(figsize=(16, 12))

# График 1: Заказы по дням недели
plt.subplot(2, 2, 1)
sns.barplot(
x='day_name',
y='total_orders',
data=fall_2024_df.groupby('day_name').agg(total_orders=('order_id', 'nunique')).reset_index(),
order=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'],
palette='viridis'
)
plt.title('Количество заказов по дням недели')
plt.xticks(rotation=45)


# In[57]:


# График 2: DAU по дням недели
plt.figure(figsize=(16, 12))
plt.subplot(2, 2, 2)
sns.barplot(
x='day_name',
y='unique_users',
data=fall_2024_df.groupby('day_name').agg(unique_users=('user_id', 'nunique')).reset_index(),
order=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'],
palette='plasma'
)
plt.title('DAU по дням недели')
plt.xticks(rotation=45)


# In[58]:


# График 3: Сравнение активности будни/выходные
plt.figure(figsize=(16, 12))
plt.subplot(2, 2, 3)
sns.barplot(
x='is_weekend',
y='total_orders',
data=weekend_metrics,
palette='magma'
)
plt.title('Сравнение активности: будни vs выходные')
plt.xticks(rotation=0)


# In[59]:


# График 4: Среднее число заказов на пользователя
plt.figure(figsize=(16, 12))
plt.subplot(2, 2, 4)
sns.barplot(
x='day_name',
y='avg_orders_per_user',
data=weekly_metrics.merge(
fall_2024_df[['day_of_week', 'day_name']].drop_duplicates(),
on='day_of_week'
),
order=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'],
palette='coolwarm')


# ВЫВОД: По будням чаще покупаются билеты. Самый активный день по количеству заказов вторник, а самый высокий DAU в четверг. 

# In[60]:


required_columns = ['region_name', 'event_id', 'order_id']
if not all(col in mdf.columns for col in required_columns):
    raise ValueError(f"Отсутствуют необходимые столбцы: {required_columns}")
mdf['event_id'] = pd.to_numeric(mdf['event_id'], errors='coerce')
mdf['order_id'] = pd.to_numeric(mdf['order_id'], errors='coerce')
mdf.dropna(subset=['region_name', 'event_id', 'order_id'], inplace=True)

region_metrics = mdf.groupby('region_name').agg(
    unique_events=('event_id', 'nunique'),
    total_orders=('order_id', 'nunique')
).reset_index()


# In[61]:


result_table = region_metrics.copy()
result_table.columns = ['Регион', 'Уникальных мероприятий', 'Всего заказов']
top_10_regions = result_table.sort_values('Уникальных мероприятий', ascending=False).head(10)
top_10_regions['Уникальных мероприятий'] = top_10_regions['Уникальных мероприятий'].astype(int)
top_10_regions['Всего заказов'] = top_10_regions['Всего заказов'].astype(int)
print("Топ-10 регионов по количеству мероприятий:")
display(top_10_regions.style
    .format({'Уникальных мероприятий': '{:,}', 'Всего заказов': '{:,}'})
    .background_gradient(cmap='viridis', subset='Уникальных мероприятий')
    .highlight_max(subset='Уникальных мероприятий', color='lightgreen')
    .set_caption('Топ-10 регионов по мероприятиям и заказам'))


# In[62]:


print("\nОбщая статистика:")
print(f"Всего регионов: {result_table.shape[0]}")
print(f"Максимальное количество мероприятий в регионе: {result_table['Уникальных мероприятий'].max()}")
print(f"Среднее количество мероприятий на регион: {result_table['Уникальных мероприятий'].mean():.2f}")
print("\nСтатистика топ-10 регионов:")
print(f"Среднее количество мероприятий в топ-10: {top_10_regions['Уникальных мероприятий'].mean():.2f}")
print(f"Общее количество заказов в топ-10: {top_10_regions['Всего заказов'].sum():,}")


# Составлен топ-10 регионов по количеству мероприятий. Самое большое количество в Каменевском регионе 5625шт

# In[63]:


# Анализ партнёров
partner_metrics = fall_2024_df.groupby('service_name').agg(
    unique_events=('event_id', 'nunique'),  # Уникальные мероприятия
    total_orders=('order_id', 'nunique'),   # Общее количество заказов
    total_revenue=('revenue_rub', 'sum'),   # Общая выручка
    unique_users=('user_id', 'nunique')     # Уникальные пользователи
).reset_index()

# Преобразуем выручку в числовой формат
partner_metrics['total_revenue'] = pd.to_numeric(partner_metrics['total_revenue'], errors='coerce')

# Сортируем по выручке
partner_metrics = partner_metrics.sort_values('total_revenue', ascending=False)

# Рассчитываем дополнительные метрики
partner_metrics['avg_orders_per_event'] = partner_metrics['total_orders'] / partner_metrics['unique_events']
partner_metrics['avg_revenue_per_order'] = partner_metrics['total_revenue'] / partner_metrics['total_orders']

# Формируем таблицу с основными метриками
basic_partner_table = partner_metrics[['service_name', 'unique_events', 'total_orders', 
                                      'total_revenue', 'unique_users']].head(10).sort_values('total_revenue', ascending=False)

basic_partner_table.columns = ['Партнёр', 'Уникальных мероприятий', 
                              'Всего заказов', 'Общая выручка (руб.)', 'Уникальных пользователей']

# Форматирование чисел
basic_partner_table['Общая выручка (руб.)'] = basic_partner_table['Общая выручка (руб.)'].apply(lambda x: f"{x:,.0f}".replace(',', ' '))

# Создаем копию для визуализации без форматирования
basic_partner_table_numeric = basic_partner_table.copy()
basic_partner_table_numeric['Общая выручка (руб.)'] = basic_partner_table_numeric['Общая выручка (руб.)'].str.replace(' ', '').astype(float)

# Таблица с расчётными метриками
calculated_partner_table = partner_metrics[['service_name', 'avg_orders_per_event', 
                                           'avg_revenue_per_order']].head(10).sort_values('avg_orders_per_event', ascending=False)

calculated_partner_table.columns = ['Партнёр', 'Заказов на мероприятие', 
                                   'Средняя выручка с заказа']

# Форматирование расчётных метрик
calculated_partner_table['Заказов на мероприятие'] = calculated_partner_table['Заказов на мероприятие'].apply(lambda x: f"{x:.2f}")
calculated_partner_table['Средняя выручка с заказа'] = calculated_partner_table['Средняя выручка с заказа'].apply(lambda x: f"{x:,.2f}".replace(',', ' '))


# Вывод таблиц
print("Таблица 1. Основные метрики по партнёрам")
display(basic_partner_table.style.format({'Уникальных мероприятий': '{:,}',
                                         'Всего заказов': '{:,}',
                                         'Уникальных пользователей': '{:,}'
                                         }))

print("\nТаблица 2. Расчётные метрики по партнёрам")
display(calculated_partner_table)


# In[64]:


top_partner_revenue = partner_metrics[['service_name', 'total_revenue']]     .sort_values('total_revenue', ascending=False)     .head(10)     .reset_index(drop=True)

total_revenue = top_partner_revenue['total_revenue'].sum()
sns.set(style='whitegrid')
plt.figure(figsize=(14, 8))
order = top_partner_revenue['service_name']
ax = sns.barplot(
    x='total_revenue',
    y='service_name',
    data=top_partner_revenue,
    order=order,
    palette='viridis',
    saturation=0.7)
for i, row in top_partner_revenue.iterrows():
    revenue = row['total_revenue']
    percentage = (revenue / total_revenue) * 100
    formatted_revenue = f"{revenue:,.0f}".replace(',', ' ')  # Форматируем число
    formatted_percentage = f"{percentage:.1f}%"
    
    ax.text(
        x=revenue + (revenue * 0.02),  # Динамическое смещение
        y=i,
        s=f"{formatted_revenue} ({formatted_percentage})",
        ha='left',
        va='center',
        fontsize=10,
        color='black'
    )
plt.title('Топ-10 партнёров по выручке', fontsize=16)
plt.xlabel('Общая выручка (руб.)', fontsize=14)
plt.ylabel('Партнёр', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.gca().xaxis.set_major_formatter('{:,.0f}'.format)
plt.axvline(x=total_revenue, color='red', linestyle='--', linewidth=1)
plt.text(
    x=total_revenue + 100000,
    y=10,
    s=f"Общая выручка: {total_revenue:,.0f} руб.",
    color='red',
    fontsize=12
)
plt.tight_layout()
plt.show()


# In[65]:


top_partner_orders = partner_metrics[['service_name', 'total_orders']]     .sort_values('total_orders', ascending=False)     .head(10)     .reset_index(drop=True)
sns.set(style='whitegrid')
plt.figure(figsize=(12, 6))
order = top_partner_orders['service_name']
ax = sns.barplot(
    x='total_orders',
    y='service_name',
    data=top_partner_orders,
    order=order,
    palette='viridis',
    saturation=0.7)

for i, row in top_partner_orders.iterrows():
    orders = row['total_orders']
    formatted_orders = f"{orders:,.0f}".replace(',', ' ')  # Форматируем число
    ax.text(
        x=orders + 150,  # Смещение текста
        y=i,
        s=formatted_orders,
        ha='left',
        va='center',
        fontsize=10,
        color='black'
    )


plt.title('Топ-10 партнёров по количеству заказов', fontsize=16)
plt.xlabel('Количество заказов', fontsize=14)
plt.ylabel('Партнёр', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.gca().xaxis.set_major_formatter('{:,.0f}'.format)
plt.tight_layout()
plt.show()


# In[66]:


top_partner_events = partner_metrics[['service_name', 'unique_events']]     .sort_values('unique_events', ascending=False)     .head(10)     .reset_index(drop=True)
sns.set(style='whitegrid')
plt.figure(figsize=(12, 6))
order = top_partner_events['service_name']
ax = sns.barplot(
    x='unique_events',
    y='service_name',
    data=top_partner_events,
    order=order,
    palette='viridis',
    saturation=0.7
)
for i, row in top_partner_events.iterrows():
    events = row['unique_events']
    formatted_events = f"{events:,.0f}".replace(',', ' ')  
    ax.text(
        x=events + 150,  
        y=i,
        s=formatted_events,
        ha='left',
        va='center',
        fontsize=10,
        color='black'
    )
plt.title('Топ-10 партнёров по количеству мероприятий', fontsize=16)
plt.xlabel('Количество мероприятий', fontsize=14)
plt.ylabel('Партнёр', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.gca().xaxis.set_major_formatter('{:,.0f}'.format)
plt.tight_layout()
plt.show()


# In[67]:


partner_metrics['avg_revenue'] = partner_metrics['total_revenue'] / partner_metrics['total_orders']
top_partner_avg_revenue = partner_metrics[['service_name', 'avg_revenue']]     .sort_values('avg_revenue', ascending=False)     .head(10)     .reset_index(drop=True)
sns.set(style='whitegrid')
plt.figure(figsize=(14, 8))
order = top_partner_avg_revenue['service_name']
ax = sns.barplot(
    x='avg_revenue',
    y='service_name',
    data=top_partner_avg_revenue,
    order=order,
    palette='viridis',
    saturation=0.7
)
for i, row in top_partner_avg_revenue.iterrows():
    avg_revenue = row['avg_revenue']
    formatted_revenue = f"{avg_revenue:,.2f}".replace(',', ' ')  # Форматируем число с 2 знаками после запятой
    
    ax.text(
        x=avg_revenue + (avg_revenue * 0.05),  # Динамическое смещение
        y=i,
        s=f"{formatted_revenue} руб.",
        ha='left',
        va='center',
        fontsize=10,
        color='black'
    )
plt.title('Топ-10 партнёров по средней выручке', fontsize=16)
plt.xlabel('Средняя выручка на заказ (руб.)', fontsize=14)
plt.ylabel('Партнёр', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.gca().xaxis.set_major_formatter('{:,.2f}'.format)
plt.tight_layout()
plt.show()


# ВЫВОД: В топе Каменевский регион, Североярская область. По выручке в лидирует Билеты без проблем, по количеству мероприятий Лови билет, а по количеству заказов Билеты без проблем. 

# ### Статистический анализ данных

# Формулирование гипотез: 
# Нулевая гипотеза (H0):
# Среднее количество заказов на одного пользователя мобильного приложения равно или меньше, чем у пользователей стационарных устройств.
# H0:μмобильное ≤ μ стационарное
# 
# Альтернативная гипотеза (H1):
# Среднее количество заказов на одного пользователя мобильного приложения выше, чем у пользователей стационарных устройств.
# H1:μмобильное > μстационарное
# 

# In[68]:


user_group_counts = mdf.groupby('user_id')['device_type_canonical'].nunique()
cleaned_mdf = mdf[mdf['user_id'].isin(user_group_counts[user_group_counts == 1].index)]

print(f"Удалено {len(mdf) - len(cleaned_mdf)} записей пересекающихся пользователей")

sum_orders = cleaned_mdf.groupby('device_type_canonical')['order_id'].nunique()

plt.figure(figsize=(10, 10))
plt.pie(
    sum_orders,
    labels=sum_orders.index,
    autopct='%1.1f%%',
    startangle=140,
    colors=plt.cm.tab20.colors,
    wedgeprops={'linewidth': 1, 'edgecolor': 'white'}
)
plt.title('Распределение общего количества заказов по типу устройства', fontsize=14)
plt.axis('equal')
plt.tight_layout()
plt.show()


# In[69]:


grouped = cleaned_mdf.groupby('device_type_canonical')['order_id'].agg(['mean', 'count', 'std', 'min', 'max'])
display(grouped)
device_describe = mdf.groupby('device_type_canonical')['order_id'].describe()
display(device_describe)


# In[70]:


c =cleaned_mdf.groupby('device_type_canonical')['order_id'].count()

plt.figure(figsize=(8, 6))
c.plot(kind='bar', color='skyblue')
plt.title('Количество количество заказов по усройствам')
plt.ylabel('количество заказов')
plt.xlabel('Устройство')
plt.show()


# In[71]:


plt.figure(figsize=(10,6))
grouped['mean'].plot(kind='bar', color='skyblue')
plt.xlabel('устройство')
plt.ylabel('Среднее количество заказов')
plt.title('Среднее  количество заказов по устройствам')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# In[72]:


desktop = cleaned_mdf[cleaned_mdf['device_type_canonical'] == 'desktop']['order_id']
mobile = cleaned_mdf[cleaned_mdf['device_type_canonical'] == 'mobile']['order_id']


# In[73]:



plt.figure(figsize=(14, 12))
plt.subplot(2, 2, 1)
sns.histplot(desktop, kde=True, color='blue', alpha=0.6, label='Desktop')
sns.histplot(mobile, kde=True, color='orange', alpha=0.6, label='Mobile')
plt.title('Распределение количества заказов')
plt.legend()


# In[74]:


plt.figure(figsize=(14, 12))
plt.subplot(2, 2, 2)
sns.boxplot(data=cleaned_mdf, x='device_type_canonical', y='order_id')
plt.title('Boxplot распределения')


# In[75]:


plt.figure(figsize=(14, 12))

plt.subplot(2, 2, 3)
stats.probplot(desktop, plot=plt)
plt.title('Q-Q plot для Desktop')


# In[76]:


plt.figure(figsize=(14, 12))
plt.subplot(2, 2, 4)
stats.probplot(mobile, plot=plt)
plt.title('Q-Q plot для Mobile')

plt.tight_layout()
plt.show()


# In[77]:



print("Тест Шапиро-Уилка на нормальность:")
print(f"Desktop: {stats.shapiro(desktop)}")
print(f"Mobile: {stats.shapiro(mobile)}")


# In[78]:


levene_test = stats.levene(desktop, mobile)
print(f"\nТест Левена на равенство дисперсий: p-value = {levene_test.pvalue}")


# In[79]:


# Формулировка гипотез
# H0: μ_mobile ≤ μ_desktop
# H1: μ_mobile > μ_desktop

alpha = 0.05


if len(desktop) < 2 or len(mobile) < 2:
    print("\nНедостаточно данных для проведения теста.")
else:
    
    res = stats.ttest_ind(mobile, desktop, equal_var=False)
    
    
    t_stat = res.statistic
    p_value_two_sided = res.pvalue
    p_value_one_sided = p_value_two_sided / 2
    
    print("\nРезультаты статистического теста:")
    print(f"T-статистика: {t_stat:.4f}")
    print(f"P-value (двусторонний): {p_value_two_sided:.4f}")
    print(f"P-value (односторонний): {p_value_one_sided:.4f}")
    
    if p_value_one_sided < alpha:
        print("\nРезультат:")
        print("Отвергаем нулевую гипотезу.")
        print("Есть статистически значимые доказательства того, что среднее количество заказов в мобильном приложении выше, чем на стационарных устройствах.")
    else:
        print("\nРезультат:")
        print("Нет оснований отвергать нулевую гипотезу.")
        print("Данные не позволяют утверждать, что среднее количество заказов в мобильном приложении статистически значимо выше, чем на стационарных устройствах.")

mean_mobile = mobile.mean()
mean_desktop = desktop.mean()
print(f"\nСреднее количество заказов:")
print(f"Мобильное: {mean_mobile:.2f}")
print(f"Стационарное: {mean_desktop:.2f}")


# Выбор scipy.stats.ttest_ind() с equal_var=False (Welch's t-test) — это лучший вариант для этого анализа потому, что он:
# 
# Не требует равенства дисперсий,
# Работает с разными размерностями групп,
# Обеспечивает надежные результаты при неравномерных данных,
# Позволяет тестировать гипотезы о том, что среднее в одной группе больше другого.

# Нулевая гипотеза (H₀):
# Среднее время между заказами пользователей мобильных приложений равно среднему времени между заказами пользователей стационарных устройств.H₀: μ_mobile ≤ μ_desktop
# 
# Альтернативная гипотеза (H₁):
# Среднее время между заказами пользователей мобильных приложений выше по сравнению с пользователями стационарных устройств.
# H₁: μ_mobile > μ_desktop

# In[80]:


pivot_table = cleaned_mdf.pivot_table(
    index='device_type_canonical',                    
    values='days_since_prev',                  
    aggfunc=['sum', 'mean']            
)

print(pivot_table)


# In[81]:


sum_hours = cleaned_mdf.groupby('device_type_canonical')['days_since_prev'].sum()

# Построение круговой диаграммы для суммы часов по городам
plt.figure(figsize=(8, 8))
plt.pie(sum_hours, labels=sum_hours.index, autopct='%1.1f%%', startangle=140)
plt.title('Распределение общего времени активности по городам (сумма часов)')
plt.show()

# Группируем по городам: расчет среднего времени
mean_hours = cleaned_mdf.groupby('device_type_canonical')['days_since_prev'].mean()

# Построение круговой диаграммы для среднего времени по городам
plt.figure(figsize=(8, 8))
plt.pie(mean_hours, labels=mean_hours.index, autopct='%1.1f%%', startangle=140)
plt.title('Распределение среднего времени активности по городам')
plt.show()


# In[82]:


grouped = cleaned_mdf.groupby('device_type_canonical')['days_since_prev'].agg(['sum', 'mean', 'count', 'std', 'min', 'max'])
display(grouped)
describe_by_city = cleaned_mdf.groupby('device_type_canonical')['days_since_prev'].describe()
display(describe_by_city)


# In[83]:


count_hours = cleaned_mdf.groupby('device_type_canonical')['days_since_prev'].count()

plt.figure(figsize=(8, 6))
count_hours.plot(kind='bar', color='skyblue')
plt.title('Количество часов по устройствам')
plt.ylabel('Число часов')
plt.xlabel('Устройство')
plt.show()


# In[84]:


plt.figure(figsize=(10,6))
grouped['mean'].plot(kind='bar', color='skyblue')
plt.xlabel('Устройство')
plt.ylabel('Среднее время между заказами')
plt.title('Среднее время между заказами по устройствам')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# In[85]:


desktop = cleaned_mdf[cleaned_mdf['device_type_canonical'] == 'desktop']['days_since_prev'].dropna()
mobile = cleaned_mdf[cleaned_mdf['device_type_canonical'] == 'mobile']['days_since_prev'].dropna()


# In[86]:


plt.figure(figsize=(14, 12))

plt.subplot(2, 2, 1)
sns.histplot(desktop, kde=True, color='blue', alpha=0.6, label='Desktop')
sns.histplot(mobile, kde=True, color='orange', alpha=0.6, label='Mobile')
plt.title('Распределение времени между заказами')
plt.xlabel('Дни между заказами')
plt.legend()


# In[87]:


plt.figure(figsize=(14, 12))

plt.subplot(2, 2, 2)
sns.boxplot(data=cleaned_mdf, x='device_type_canonical', y='days_since_prev')
plt.title('Boxplot распределения времени между заказами')
plt.ylabel('Дни между заказами')


# In[88]:


plt.figure(figsize=(14, 12))
plt.subplot(2, 2, 3)
stats.probplot(desktop, plot=plt)
plt.title('Q-Q plot для Desktop')


# In[89]:


plt.figure(figsize=(14, 12))
plt.subplot(2, 2, 4)
stats.probplot(mobile, plot=plt)
plt.title('Q-Q plot для Mobile')

plt.tight_layout()
plt.show()


# In[90]:



print("Тест Шапиро-Уилка на нормальность:")
print(f"Desktop: {stats.shapiro(desktop)}")
print(f"Mobile: {stats.shapiro(mobile)}")

levene_test = stats.levene(desktop, mobile)
print(f"\nТест Левена на равенство дисперсий: p-value = {levene_test.pvalue}")


# In[91]:



# Формулировка гипотез
# H0: μ_mobile ≤ μ_desktop
# H1: μ_mobile > μ_desktop

alpha = 0.05

if len(desktop) < 2 or len(mobile) < 2:
    print("\nНедостаточно данных для проведения теста.")
else:
    # Проверяем нормальность распределения
    if (stats.shapiro(desktop).pvalue > 0.05) and (stats.shapiro(mobile).pvalue > 0.05):
        # Если данные нормально распределены
        res = stats.ttest_ind(mobile, desktop, equal_var=False, alternative='greater')
    else:
        # Если данные не нормально распределены
        res = stats.mannwhitneyu(mobile, desktop, alternative='greater')
    
    # Получаем статистику и p-value
    statistic = res.statistic
    p_value = res.pvalue
    
    print("\nРезультаты статистического теста:")
    print(f"Статистика теста: {statistic:.4f}")
    print(f"P-value: {p_value:.4f}")
    
    if p_value < alpha:
        print("\nРезультат:")
        print("Отвергаем нулевую гипотезу.")
        print("Есть статистически значимые доказательства того, что среднее время между заказами в мобильном приложении больше, чем на стационарных устройствах.")
    else:
        print("\nРезультат:")
        print("Нет оснований отвергать нулевую гипотезу.")
        print("Данные не позволяют утверждать, что среднее время между заказами в мобильном приложении статистически значимо больше, чем на стационарных устройствах.")

mean_mobile = mobile.mean()
mean_desktop = desktop.mean()
print(f"\nСреднее время между заказами:")
print(f"Мобильное: {mean_mobile:.2f} дней")
print(f"Стационарное: {mean_desktop:.2f} дней")


# ### Общий вывод и рекомендации

# В данном анализе использовались данные трех разных датасетов: final_tickets_orders_df, final_tickets_events_df и final_tickets_tenge_df. Данные были обработаны на предмет дубликатов и пропусков, а также объеденены в один датафрейм mdf. Типы данных были приведены к нормальным для анализа. Данных достаточно для анализа. 
# 
# Исходя их полученных результатов можно сказать, что самыми пополярными мероприятиями являются другое, концерты. Причем осенью лидируют эти же категории, но заказов в них уже меньше. Количество заказов в целом осенью больше, а вот средний чек осенью ниже. Среди площадок есть определенные лидеры по выручке: Билеты без проблем; по количеству мероприятий: Лови билет; по количеству заказов: Билеты без проблем, Билеты в руки и Лови билет. Среди регионов в топе Каменевский регион и Североярская область. 
# 
# Исходя из проверки гипотез можно с уверенностью сказать, что среднее количество билетов больше продается через мобильное приложение. Нулевая гипотеза была опровергнута. Стоит далее развивать мобильное приложение и ввести какую-либо рекламную кампанию для пользователей мобильного приложения. В целом результат объясним тем, что существует общемировая тенденция ухода от пк в пользу мобильных приложений, в случаях, если пк не нужен для работы.
# 
# Также проверена гипотеза Среднее время между заказами пользователей мобильных приложений выше по сравнению с пользователями стационарных устройств. Данная теория была опровергнута. Данные показали что среднее время между заказами выше у мобильных устройств.
