#!/usr/bin/env python
# coding: utf-8

# # Часть 1. Проверка гипотезы в Python и составление аналитической записки

# Вы предобработали данные в SQL, и теперь они готовы для проверки гипотезы в Python. Загрузите данные пользователей из Москвы и Санкт-Петербурга c суммой часов их активности из файла yandex_knigi_data.csv. Если работаете локально, скачать файл можно по ссылке.
# 
# Проверьте наличие дубликатов в идентификаторах пользователей. Сравните размеры групп, их статистики и распределение.
# 
# Напомним, как выглядит гипотеза: пользователи из Санкт-Петербурга проводят в среднем больше времени за чтением и прослушиванием книг в приложении, чем пользователи из Москвы. Попробуйте статистически это доказать, используя одностороннюю проверку гипотезы с двумя выборками:
# 
# Нулевая гипотеза $H_0: \mu_{\text{СПб}} \leq \mu_{\text{Москва}}$ <br> Среднее время активности пользователей в Санкт-Петербурге не больше, чем в Москве.
# 
# Альтернативная гипотеза $H_1: \mu_{\text{СПб}} > \mu_{\text{Москва}}$ <br> Среднее время активности пользователей в Санкт-Петербурге больше, и это различие статистически значимо.
# 
# По результатам анализа данных подготовьте аналитическую записку, в которой опишите:
# 
# Выбранный тип t-теста и уровень статистической значимости.
# 
# Результат теста, или p-value.
# 
# Вывод на основе полученного p-value, то есть интерпретацию результатов.
# 
# Одну или две возможные причины, объясняющие полученные результаты.

# ### Проверка активности пользователей по городам
# 
# - Автор: Бусс Татьяна Сергеевна
# - Дата: 07.10 .2025

# ## Цели и задачи проекта
# 
# 1. Подготовить данные к проведению теста и провести тест.
# 2. Проанализировать результаты теста.

# ## Описание данных
# - Unnamed: 0 - порядковый номер 
# - city - город	
# - puid - идентификатор пользователя	
# - hours - активность пользователя в часах

# ## Содержимое проекта
# 
# 1. Загрузка и знакомство с данными
# 2. Подготовка данных к дальнейшему анализу и тестированию
# 3. Составление гипотез
# 4. Проверка гипотез
# 5. Оценка результатов тестирования
# 6. Выводы
# 
# ---

# ## 1. Загрузка данных и знакомство с ними
# 
# Загрузите данные пользователей из Москвы и Санкт-Петербурга c их активностью (суммой часов чтения и прослушивания) из файла `/datasets/yandex_knigi_data.csv`.

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


# In[2]:


df = pd.read_csv('https://code.s3.yandex.net//datasets/yandex_knigi_data.csv ')
display(df.head())
df.info()


# In[3]:


msno.matrix(df)
plt.title('Матрица пропусков')
plt.show()
missing_stats = pd.DataFrame({'Кол-во пропусков': df.isnull().sum(),
    'Доля пропусков': df.isnull().sum() / len(df)}).sort_values(by='Доля пропусков', ascending=False)
display(missing_stats.style.background_gradient(cmap='coolwarm'))


# In[4]:


duplicates = df.duplicated()
print(f"Количество дубликатов: {duplicates.sum()}")
df_clean = df.drop_duplicates()
print(f"Количество строк после удаления дубликатов: {len(df_clean)}")


# In[5]:


df['puid'].value_counts().head()


# In[6]:


check_puid = df.groupby('puid').agg(un_cities = ('city', 'nunique'), cities = ('city', 'count'))
check_puid.query('un_cities != cities')


# In[8]:


df_unique_puid = df.drop_duplicates(subset=['puid'])
check_puid = df.groupby('puid').agg(
    un_cities=('city', 'nunique'),
    cities=('city', 'count'))
result = check_puid.query('un_cities != 1')
print(result)


# Данные проверены на дубликаты и пропуски, а также проверены типы данных. 

# ## 2. Проверка гипотезы в Python
# 
# Гипотеза звучит так: пользователи из Санкт-Петербурга проводят в среднем больше времени за чтением и прослушиванием книг в приложении, чем пользователи из Москвы. Попробуйте статистически это доказать, используя одностороннюю проверку гипотезы с двумя выборками:
# 
# - Нулевая гипотеза H₀: Средняя активность пользователей в часах в двух группах (Москва и Санкт-Петербург) не различается.
# 
# - Альтернативная гипотеза H₁: Средняя активность пользователей в Санкт-Петербурге больше, и это различие статистически значимо.

# In[13]:


pivot_table = df_unique_puid.pivot_table(
    index='city',                    
    values='hours',                  
    aggfunc=['sum', 'mean']            
)

print(pivot_table)


# In[14]:


sum_hours = df_unique_puid.groupby('city')['hours'].sum()

# Построение круговой диаграммы для суммы часов по городам
plt.figure(figsize=(8, 8))
plt.pie(sum_hours, labels=sum_hours.index, autopct='%1.1f%%', startangle=140)
plt.title('Распределение общего времени активности по городам (сумма часов)')
plt.show()

# Группируем по городам: расчет среднего времени
mean_hours = df.groupby('city')['hours'].mean()

# Построение круговой диаграммы для среднего времени по городам
plt.figure(figsize=(8, 8))
plt.pie(mean_hours, labels=mean_hours.index, autopct='%1.1f%%', startangle=140)
plt.title('Распределение среднего времени активности по городам')
plt.show()


# In[16]:


grouped = df_unique_puid.groupby('city')['hours'].agg(['sum', 'mean', 'count', 'std', 'min', 'max'])
display(grouped)
describe_by_city = df_unique_puid.groupby('city')['hours'].describe()
display(describe_by_city)


# In[17]:


count_hours = df_unique_puid.groupby('city')['hours'].count()

plt.figure(figsize=(8, 6))
count_hours.plot(kind='bar', color='skyblue')
plt.title('Количество записей по городам')
plt.ylabel('Число записей')
plt.xlabel('Город')
plt.show()


# In[18]:


plt.figure(figsize=(10,6))
grouped['mean'].plot(kind='bar', color='skyblue')
plt.xlabel('Город')
plt.ylabel('Среднее время активности (hours)')
plt.title('Среднее время активности по городам')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# По графикам видим, что количество записей в Москве больше, при этом среднее количество прослушиваний в Санкт-Петербурге выше. 

# In[19]:


moscow_hours = df_unique_puid[df_unique_puid['city'] == 'Москва']['hours']
spb_hours = df_unique_puid[df_unique_puid['city'] == 'Санкт-Петербург']['hours']

alpha = 0.05

if len(moscow_hours) < 2 or len(spb_hours) < 2:
    print("Недостаточно данных для проведения теста.")
else:
    # Welch's t-test
    res = stats.ttest_ind(spb_hours, moscow_hours, equal_var=False)

    t_stat = res.statistic
    print(f't-statistic: {t_stat:.4f}')
    print(f'p-value (двусторонний): {res.pvalue:.4f}')

    # Односторонний p-value для гипотезы "СПБ > Москва"
    if t_stat > 0:
        p_value_one_side = res.pvalue / 2
    else:
        p_value_one_side = 1 - res.pvalue / 2

    print(f'p-value (односторонний): {p_value_one_side:.4f}')

    if p_value_one_side < alpha:
        print("Статистически значимо: средняя активность в Санкт-Петербурге больше, чем в Москве.")
    else:
        print("Нет оснований утверждать, что средняя активность в СПБ больше, чем в Москве.")
        print("Нет оснований отвергать нулевую гипотезу: данные не подтверждают, что средняя активность из СПБ меньше, чем из Москвы.")


# Выбор scipy.stats.ttest_ind() с equal_var=False (Welch's t-test) — это лучший вариант для этого анализа потому, что он:
# 
# - Не требует равенства дисперсий,
# - Работает с разными размерностями групп,
# - Обеспечивает надежные результаты при неравномерных данных,
# - Позволяет тестировать гипотезы о том, что среднее в одной группе больше другого.

# ## 3. Аналитическая записка
# По результатам анализа данных подготовьте аналитическую записку, в которой опишете:
# 
# - Выбранный тип t-теста и уровень статистической значимости.
# 
# - Результат теста, или p-value.
# 
# - Вывод на основе полученного p-value, то есть интерпретацию результатов.
# 
# - Одну или две возможные причины, объясняющие полученные результаты.
# 
# 

# Для проверки гипотезы о различии средней активности пользователей из двух групп — Москвы и Санкт-Петербурга — был применён независимый двухвыборочный t-тест Уэлча (Welch’s t-test). Этот тест выбран потому, что группы неравномерных размеров.
# Уровень статистической значимости установлен на стандартном уровне α = 0.05.
# Значение t-статистики: 0.4028
# Двустороннее p-value: 0.6871
# Одностороннее p-value (исходя из гипотезы, что активность в СПБ больше, чем в Москве): 0.3436
# Полученное одностороннее p-value (0.3436) существенно превышает установленный уровень значимости (0.05). Это означает, что отсутствуют статистически значимые доказательства того, что средняя активность пользователей из Санкт-Петербурга выше, чем из Москвы.
# 
# Таким образом, нулевая гипотеза о равенстве средней активности не отвергается.
# В Москве больше читают. Связано это может быть с тем, что в Москве больше население примерно в 2 раза. Ещё одной причиной может быть то, что в Москве больше финансовых возможностей для активного использования подписок на чтение. 

# ----

# # Часть 2. Анализ результатов A/B-тестирования

# Теперь вам нужно проанализировать другие данные. Представьте, что к вам обратились представители интернет-магазина BitMotion Kit, в котором продаются геймифицированные товары для тех, кто ведёт здоровый образ жизни. У него есть своя целевая аудитория, даже появились хиты продаж: эспандер со счётчиком и напоминанием, так и подстольный велотренажёр с Bluetooth.
# 
# В будущем компания хочет расширить ассортимент товаров. Но перед этим нужно решить одну проблему. Интерфейс онлайн-магазина слишком сложен для пользователей — об этом говорят отзывы.
# 
# Чтобы привлечь новых клиентов и увеличить число продаж, владельцы магазина разработали новую версию сайта и протестировали его на части пользователей. По задумке, это решение доказуемо повысит количество пользователей, которые совершат покупку.
# 
# Ваша задача — провести оценку результатов A/B-теста. В вашем распоряжении:
# 
# * данные о действиях пользователей и распределении их на группы,
# 
# * техническое задание.
# 
# Оцените корректность проведения теста и проанализируйте его результаты.

# ## 1. Опишите цели исследования.
# 
# 

# Провести оценку изменения интерфейса сайта.

# ## 2. Загрузите данные, оцените их целостность.
# 

# In[20]:


# Загрузка csv напрямую
url_p = 'https://code.s3.yandex.net/datasets/ab_test_participants.csv'
response_p = requests.get(url_p)
p = pd.read_csv(StringIO(response_p.text))

# Загрузка zip архива и извлечение нужного csv
url_e = 'https://code.s3.yandex.net/datasets/ab_test_events.zip'
response_e = requests.get(url_e)
with zipfile.ZipFile(BytesIO(response_e.content)) as z:
    # Предположим, в архиве один файл — можно получить имен файла так:
    filename = z.namelist()[0]
    with z.open(filename) as f:
        e = pd.read_csv(f, parse_dates=['event_dt'], low_memory=False)


# In[21]:


df= pd.merge(p, e, on='user_id', how='left')


# In[22]:


display(df.head())
df.info()


# In[23]:


msno.matrix(df)
plt.title('Матрица пропусков')
plt.show()
missing_stats = pd.DataFrame({'Кол-во пропусков': p.isnull().sum(),
    'Доля пропусков': df.isnull().sum() / len(df)}).sort_values(by='Доля пропусков', ascending=False)
display(missing_stats.style.background_gradient(cmap='coolwarm'))


# In[24]:


duplicates = df.duplicated()
print(f"Количество дубликатов: {duplicates.sum()}")
df_clean = df.drop_duplicates()
print(f"Количество строк после удаления дубликатов: {len(df_clean)}")


# ## 3. По таблице `ab_test_participants` оцените корректность проведения теста:
# 
#    3\.1 Выделите пользователей, участвующих в тесте, и проверьте:
# 
#    - соответствие требованиям технического задания,
# 
#    - равномерность распределения пользователей по группам теста,
# 
#    - отсутствие пересечений с конкурирующим тестом (нет пользователей, участвующих одновременно в двух тестовых группах).

# 
# https://code.s3.yandex.net/datasets/ab_test_participants.csv — таблица участников тестов.
# 
# Структура файла:
# 
# - user_id — идентификатор пользователя;
# 
# - group — группа пользователя;
# 
# - ab_test — название теста;
# 
# - device — устройство, с которого происходила регистрация.
# 
# https://code.s3.yandex.net/datasets/ab_test_events.zip — архив с одним csv-файлом, в котором собраны события 2020 года;
# 
# Структура файла:
# 
# - user_id — идентификатор пользователя;
# 
# - event_dt — дата и время события;
# 
# - event_name — тип события;
# 
# - details — дополнительные данные о событии.

# 3\.2 Проанализируйте данные о пользовательской активности по таблице `ab_test_events`:
# 
# - оставьте только события, связанные с участвующими в изучаемом тесте пользователями;

# In[25]:


test_name = 'interface_eu_test'
df = df_clean[df_clean['ab_test'] == test_name]


# In[26]:


num_users = df['user_id'].nunique()
print("Число уникальных участников теста:", num_users)
group_user_counts = df.groupby('group')['user_id'].nunique()
print("Распределение участников по группам:\n", group_user_counts)
print("Статистика распределения участников по группам:\n", group_user_counts.describe())


# In[27]:


df_b = df_clean[df_clean['group'] == 'B']
user_tests_in_B = df_b.groupby('user_id')['ab_test'].nunique()
conflicting_users_in_B = user_tests_in_B[user_tests_in_B > 1].index
print("Пользователи, попавшие в группу В в нескольких тестах одновременно:", len(conflicting_users_in_B))
df_cleaned = df[~df['user_id'].isin(conflicting_users_in_B)].copy()
print(f"Размер df до удаления: {len(df)}")
print(f"Размер df после удаления конфликтных пользователей: {len(df_cleaned)}")


# In[28]:


users_multiple_groups = df.groupby('user_id')['group'].nunique()
conflicting_groups_users = users_multiple_groups[users_multiple_groups > 1]
print("Пользователи в нескольких группах одного теста:", len(conflicting_groups_users))


# In[29]:


test_user_ids = df['user_id'].unique().tolist()
events_test = df[df['user_id'].isin(test_user_ids)]
print("Общее число событий для участников теста:", len(events_test))


# В исследуемом тесте группа A состоит из 5383 участников, а B - 5467. Группы распределены достаточно равномено, что важно для теста.

# - определите горизонт анализа: рассчитайте время (лайфтайм) совершения события пользователем после регистрации и оставьте только те события, которые были выполнены в течение первых семи дней с момента регистрации;

# In[30]:


first_dts = events_test.groupby('user_id').agg(first_dt=('event_dt', 'min')).reset_index()
events_fdt = df.merge(first_dts, on='user_id', how='left')


# In[31]:


events_fdt['lifetime'] = (pd.to_datetime(events_fdt['event_dt']) - pd.to_datetime(events_fdt['first_dt'])).dt.days
events_fdt_sorted = events_fdt.sort_values(by='lifetime')
events_7_days = events_fdt[events_fdt['lifetime'] <= 6]
display(events_7_days.head())
print(f"Общее число событий за первые 7 дней: {len(events_7_days)}")


# Оцените достаточность выборки для получения статистически значимых результатов A/B-теста. Заданные параметры:
# 
# - базовый показатель конверсии — 30%,
# 
# - мощность теста — 80%,
# 
# - достоверность теста — 95%.
# - Ожидаемый прирост - 3 п.п.

# In[32]:


baseline_conversion = 0.30   
desired_lift = 0.03          
alpha = 0.05                 
power = 0.8                  
assert 0 < baseline_conversion < 1, "Baseline конверсии должно быть между 0 и 1"
assert 0 < desired_lift < 1, "Прирост конверсии должен быть >0 и <1"

effect_size = proportion_effectsize(baseline_conversion, baseline_conversion + desired_lift)

analysis = NormalIndPower()

sample_size_per_group = analysis.solve_power(
    -effect_size,
    power=power,
    alpha=alpha,
    alternative='larger'  # односторонний тест
)

print(f"Минимальный размер выборки на группу: {int(sample_size_per_group)} пользователей")


# Минимальный размер выборки 2963 пользователей. Общее число событий за первые 7 дней: 63449. Данных достаточно для проведения теста

# - рассчитайте для каждой группы количество посетителей, сделавших покупку, и общее количество посетителей.

# In[33]:


users_in_group = events_7_days.groupby('group')['user_id'].nunique()

purchases_in_group = events_7_days[events_7_days['event_name'] == 'purchase'].groupby('group')['user_id'].nunique()

for grp in users_in_group.index:
    total_users = users_in_group[grp]
    buy_users = purchases_in_group.get(grp, 0) 
    print(f"Группа: {grp}")
    print(f" - Общее число уникальных пользователей: {total_users}")
    print(f" - Число пользователей, сделавших покупку: {buy_users}")
    print(f" - Конверсия: {buy_users / total_users:.2%}\n")


# - сделайте предварительный общий вывод об изменении пользовательской активности в тестовой группе по сравнению с контрольной.

# В текущих данных в тестовой группе конверсия выше и составляет 29.27%, конверсия базовой группы 27.49%, что говорит о возможной эффективности внедряемого нововведения. Ожидаемый прирост конверсии в 3п.п. не достигнут. Сейчас прирост составляет 1,78п.п. Для окончательного вывода о статистической значимости рекомендуются дополнительные статистические тесты.

# ## 4. Проведите оценку результатов A/B-тестирования:

# - Проверьте изменение конверсии подходящим статистическим тестом, учитывая все этапы проверки гипотез.

# Нулевая гипотеза H0: конверсия в тестовой группе равна конверсии в контрольной — изменения отсутствуют: p test=p control.
# Альтернативная гипотеза H1:p test >p control - конверсия в тестовой группе выше контрольной.

# In[37]:


from statsmodels.stats.proportion import proportion_confint
successes = events_7_days[events_7_days['event_name'] == 'purchase'].groupby('group')['user_id'].nunique()
nobs = events_7_days.groupby('group')['user_id'].nunique()
order = ['B', 'A']
successes = successes.reindex(order)
nobs = nobs.reindex(order)
assert len(successes) == len(nobs), "Длины массивов должны совпадать"
conversions = successes / nobs
print(f"Конверсии по группам:\n{conversions}")
stat, pval = sm.stats.proportions_ztest(count=successes, nobs=nobs, alternative='larger')
confint = proportion_confint(count=successes, nobs=nobs, method='normal')
print(f"Доверительные интервалы по группам:\n{confint}")
print(f"z-statistic = {stat:.3f}")
print(f"p-value (односторонний) = {pval:.4f}")
if pval < 0.05:
    print("Отвергаем H0: конверсия в тестовой группе статистически выше контрольной.")
else:
    print("Нет оснований отвергать H0: статистически значимых различий нет.")


# - Опишите выводы по проведённой оценке результатов A/B-тестирования. Что можно сказать про результаты A/B-тестирования? Был ли достигнут ожидаемый эффект в изменении конверсии?

# Проведена оценка тестирования. На данном этапе отвергаем нулевую гипотезу, так как конверсия в тестовой группе выше. Ожидаемый прирост конверсии в 3п.п. не достигнут. Сейчас конверсия только 1.78п.п. Результаты теста показывают, что нововведения привели к повышению конверсии. 
