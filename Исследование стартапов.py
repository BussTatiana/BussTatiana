#!/usr/bin/env python
# coding: utf-8

# # Проект: Исследование стартапов
# 
# Автор: Бусс Татьяна Сергеевна
# Дата: 07.07.2025

# ## Введение

# ### Цели и задачи проекта
# Подготовить датасет и проверить, что данные в нём соответствуют здравому смыслу, а также ответить на вопросы заказчика как о предобработке, так и о значении данных для бизнеса.
# 
# Задачи:
# 
# 1. Загрузить данные и познакомиться с их содержимым.
# 2. Провести предобработку данных.
# 3. Провести исследовательский анализ данных.
# 4. Сформулировать выводы по проведённому анализу

# ### Описание данных
# 
# - acquisition.csv - Содержит информацию о покупках одними компаниями других компаний
# - company_and_rounds.csv - Содержит информацию о компаниях и раундах финансирования
# - people.csv - Содержит информацию о сотрудниках
# - education.csv - Содержит информацию об образовании сотрудника
# - degrees.csv - Содержит информацию о типе образования сотрудника

# ### Содержание проекта
# 
# 1. Загрузка данных и знакомство с ними.
# 2. Предобработка данных.
# 3. Исследовательский анализ данных.
# 4. Итоговые выводы.

# In[1]:


#Подключаю необходимые библиотеки
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# 
# ## Шаг 1. Знакомство с данными: загрузка и первичная предобработка
# 
# Названия файлов:
# * acquisition.csv
# * company_and_rounds.csv
# * people.csv
# * education.csv
# * degrees.csv
# 
# Опциональные датасеты:
# * fund.csv
# * investment.csv
# 
# 
# Они находятся в папке datasets, если вы выполняете работу на платформе. В случае, если вы делаете работу локально, доступ к файлам в папке можно получить по адресу `https://code.s3.yandex.net/datasets/` + имя файла.
# 
# ### 1.1. Вывод общей информации, исправление названия столбцов
# 
# - Загрузите все данные по проекту.
# - Проверьте названия столбцов.
# - Выведите информацию, которая необходима вам для принятия решений о предобработке, для каждого из датасетов.

# In[2]:


#Загружаю датасеты
acquisition = pd.read_csv('https://code.s3.yandex.net/datasets/acquisition.csv')
company_and_rounds = pd.read_csv('https://code.s3.yandex.net/datasets/company_and_rounds.csv')
people = pd.read_csv('https://code.s3.yandex.net/datasets/people.csv')
education = pd.read_csv('https://code.s3.yandex.net/datasets/education.csv')
degrees = pd.read_csv('https://code.s3.yandex.net/datasets/degrees.csv')


# In[3]:


acquisition.head()


# In[4]:


acquisition.info()


# In[5]:


company_and_rounds.head()


# In[6]:


company_and_rounds.info()


# In[7]:


#привожу названия к snake_case
def to_snake_case(name):
    return name.replace('  ', '_')
company_and_rounds.columns = [to_snake_case(col) for col in company_and_rounds.columns]

print(company_and_rounds.columns)


# In[8]:


people.head()


# In[9]:


people.info()


# In[10]:


education.head()


# In[11]:


education.info()


# In[12]:


degrees.head()


# In[13]:


degrees.info()


# Ознакомилась с данными, вижу большое количество пропусков в датасете company_and_rounds. Их стоит рассмотреть подробнее, вероятно, пропуски образовались из-за неверного присоединения. На это же намекают и дублирующиеся названия и описания столбцов.

# ### 1.2. Смена типов и анализ пропусков
# 
# - Обработайте типы данных в столбцах, которые хранят значения даты и времени, если это необходимо.
# - Оцените полноту данных — сделайте предварительный вывод, достаточно ли данных для решения задач проекта.

# In[14]:


#вывожу типы данных
print(acquisition.dtypes, '\n', 
      company_and_rounds.dtypes, '\n', 
      people.dtypes, '\n', education.dtypes, '\n',
      degrees.dtypes)


# In[15]:


#меняю тип данных на дату там где надо
acquisition['acquired_at'] = pd.to_datetime(acquisition['acquired_at'])
company_and_rounds['founded_at'] = pd.to_datetime(company_and_rounds['founded_at'])
company_and_rounds['funded_at'] = pd.to_datetime(company_and_rounds['funded_at'])
education['graduated_at'] = pd.to_datetime(education['graduated_at'])


# In[61]:


#проверяю что типы изменились корректно
print(acquisition.dtypes, '\n', 
      company_and_rounds.dtypes, '\n', 
      people.dtypes, '\n', education.dtypes, '\n',
      degrees.dtypes)


# Предварительный вывод: Данные загружены, выведены первые строки таблиц для ознакомления. Названия столбцов приведены к единому стилю, типы данных изменены на подходящие. В данных есть пропуски. 

# In[62]:


get_ipython().system('pip install missingno')


# In[16]:


import missingno as msno
import pandas as pd
msno.matrix(company_and_rounds)
plt.title('Матрица пропусков')
plt.show()
msno.heatmap(company_and_rounds)
plt.title('Корреляция пропусков между столбцами')
plt.show()
missing_stats = pd.DataFrame({'Кол-во пропусков': company_and_rounds.isnull().sum(),
    'Доля пропусков': company_and_rounds.isnull().sum() / len(company_and_rounds)}).sort_values(by='Доля пропусков', ascending=False)
display(missing_stats.style.background_gradient(cmap='coolwarm'))


# С помощью этих визуализаций видно, что во второй части таблицы огромное количество пропусков. Вероятнее всего образовались они при не верном присоединении таблиц.

# ## Шаг 2. Предобработка данных, предварительное исследование

# 
# ### 2.1. Раунды финансирования по годам
# 
# Задание необходимо выполнить без объединения и дополнительной предобработки на основе датасета `company_and_rounds.csv`.
# 
# - Составьте сводную таблицу по годам, в которой на основании столбца `raised_amount` для каждого года указан:
#     - типичный размер средств, выделяемый в рамках одного раунда;
#     - общее количество раундов финансирования за этот год.
#     
# - Оставьте в таблице информацию только для тех лет, для которых есть информация о более чем 50 раундах финансирования.
# - На основе получившейся таблицы постройте график, который будет отражать динамику типичного размера средств, которые стартапы получали в рамках одного раунда финансирования.
# 
# На основе полученных данных ответьте на вопросы:
# 
# - В каком году типичный размер собранных в рамках одного раунда средств был максимален?
# - Какая тенденция по количеству раундов и выделяемых в рамках каждого раунда средств наблюдалась в 2013 году?

# In[17]:


#вывожу сводную таблицу по раундам финансирования
company_and_rounds['year'] = company_and_rounds['funded_at'].dt.year
pivot_table = pd.pivot_table(
    company_and_rounds,
    index='year',
    values='raised_amount',
    aggfunc=['median', 'count'])
pivot_table.columns = ['typical_raised_amount', 'number_of_rounds']
result = pivot_table[pivot_table['number_of_rounds'] > 50]

print(result)


# In[18]:


plt.figure(figsize=(10, 6))
plt.plot(result.index, result['typical_raised_amount'])
plt.title('Динамика типичного размера средств в рамках одного раунда по годам')
plt.xlabel('Год')
plt.ylabel('Типичный размер средств (медиана raised_amount)')
plt.grid(True)
plt.show()


# Типичный размер собранных в рамках одного раунда средств был максимален в 2005 году. В 2013 году наблюдается резкое увеличение количества раундов с медианной суммой финансирования 1 200 000. При этом с 2006 года медианное значение суммы финансирования плавно снижалось из-за роста количества раундов.

# 
# ### 2.2. Люди и их образование
# 
# Заказчик хочет понять, зависит ли полнота сведений о сотрудниках (например, об их образовании) от размера компаний.
# 
# - Оцените, насколько информация об образовании сотрудников полна. Используя датасеты `people.csv` и `education.csv`, разделите все компании на несколько групп по количеству сотрудников и оцените среднюю долю сотрудников без информации об образовании в каждой из групп. Обоснуйте выбранные границы групп.
# - Оцените, возможно ли для выполнения задания присоединить к этим таблицам ещё и таблицу `degrees.csv`.

# In[19]:


# объединяю датасеты и проверяю что всё ок выводом первых 5 строк
people_and_edu = pd.merge(people, education, left_on='id', right_on='person_id', how='left')
people_and_edu.head()


# In[20]:


table_people_and_edu = pd.pivot_table(people_and_edu, index = 'company_id', values = ['id_x','instituition'], aggfunc = ['count'])
table_people_and_edu.columns = ['count_people','count_education']
table_people_and_edu['share_no_education'] = 1-pd.to_numeric(table_people_and_edu['count_education'], 
                                                             downcast = 'float')/table_people_and_edu['count_people']
table_people_and_edu.head(5)


# In[21]:


bins = [0, 1, 2, 3, 5, 10, 25, float('inf')]
labels = ['1', '2', '3', '4-5', '6-10', '11-25', '26+']

company_sizes = table_people_and_edu.reset_index()
company_sizes['size_group'] = pd.cut(company_sizes['count_people'], bins=bins, labels=labels)
mean_share = company_sizes.groupby('size_group')['share_no_education'].mean()
counts = company_sizes.groupby('size_group').size()
result_df = pd.DataFrame({'average_share_no_education': mean_share,
    'company_count': counts})
print(result_df)


# Группы были выбраны таким образом, чтобы в каждой было плюс минус равное распределение по количеству сотрудников.

# В общем и целом таблица degrees дополнит данные об образовании и даст информацию о специальности сотрудника. Объединить таблицы возможно по столбцу id_x (здесь беру название из уже объединеных таблиц) и object_id. Но столбец object_id перед присоединением нужно очистить от лишних символов 'р:'. При этом для выполнения оценки наличия образования и доли сотрудников с образованием данные о специализации сотрудника не требуются.

# ### 2.3. Объединять или не объединять — вот в чём вопрос
# 
# Некоторые названия столбцов встречаются в датасетах чаще других. В результате предварительной проверки датасетов было выяснено, что столбец `company_id` подходит для объединения данных.
# 
# - Установите, подходит ли для объединения данных столбец `network_username`, который встречается в нескольких датасетах. Нам необходимо понимать, дублируется ли для разных датасетов информация в столбцах с таким названием, и если да — то насколько часто.
# - Оцените, можно ли использовать столбцы с именем `network_username` для объединения данных.

# In[22]:


set1 = set(company_and_rounds['network_username'])
set2 = set(people['network_username'])
common_usernames = set1 & set2
share_company_and_rounds = len(common_usernames) / len(set1) 
share_people = len(common_usernames) / len(set2) 
print(f"Доля названий из company_and_rounds, встречающихся в датасетах: {share_company_and_rounds:.2%}")
print(f"Доля названий из people, встречающихся в датасетах: {share_people:.2%}")


# Исходя из незначительных пересечений данный столбец не подходит для объединения датасетов.

# 
# ### 2.4. Проблемный датасет и причина возникновения пропусков
# 
# Во время собственного анализа данных у заказчика больше всего вопросов возникло к датасету `company_and_rounds.csv`. В нём много пропусков как раз в информации о раундах, которая заказчику важна.
# 
# - Любым удобным способом приведите данные в вид, который позволит в дальнейшем проводить анализ в разрезе отдельных компаний. Обратите внимание на структуру датасета, порядок и названия столбцов, проанализируйте значения.
# 
# По гипотезе заказчика данные по компаниям из этой таблицы раньше хранились иначе, более удобным для исследования образом.
# 
# - Максимальным образом сохраняя данные, сохранив их связность и исключив возможные возникающие при этом ошибки, подготовьте данные так, чтобы удобно было отобрать компании по параметрам и рассчитать показатели из расчёта на одну компанию без промежуточных агрегаций.

# In[23]:


total_rows = len(company_and_rounds)
missing_counts = company_and_rounds.isnull().sum()
missing_percentages = company_and_rounds.isnull().mean() * 100
for col in company_and_rounds.columns:
    count = missing_counts[col]
    percent = missing_percentages.loc[col]
    print(f"Столбец '{col}':")
    print(f"  Пропущенных значений: {count} ({float(percent):.2f}%)\n")


# Вероятнее всего столбцы с долей пропусков 75% появились в результате неправильного присоединения двух таблиц. Названия колонок и описание данных в них также намекает на это. Поэтому считаю что для дальнейшего анализа стоит оставить только столбцы с долей пропусков меньше 75%

# In[24]:


columns = missing_percentages[missing_percentages < 74].index.tolist()
company_and_rounds_filt = company_and_rounds[columns]
company_and_rounds_filt.head()


# In[25]:


duplicates = company_and_rounds_filt.duplicated()
print(f"Количество дубликатов: {duplicates.sum()}")
company_and_rounds_filt_clean = company_and_rounds_filt.drop_duplicates()
print(f"Количество строк после удаления дубликатов: {len(company_and_rounds_filt_clean)}")


# Теперь датасет содержит небольшое количество пропусков, очищен от дубликатов и готов к дальнейшему анализу.

# 
# ## Шаг 3. Исследовательский анализ объединённых таблиц
# 

# 
# ### 3.1. Объединение данных
# 
# Объедините данные для ответа на вопросы заказчика, которые касаются интересующих его компаний. Заказчика прежде всего интересуют те компании, которые меняли или готовы менять владельцев. Получение инвестиций или финансирования, по мнению заказчика, означает интерес к покупке или продаже компании.
# 
# В качестве основы для объединённой таблицы возьмите данные из обработанного датасета `company_and_rounds.csv` — выберите только те компании, у которых указаны значения `funding_rounds` или `investment_rounds` больше нуля, или те, у которых в колонке `status` указано `acquired`. В результирующей таблице должно получиться порядка 40 тысяч компаний.
# 
# Проверьте полноту и корректность получившейся таблицы. Далее работайте только с этими данными.

# In[26]:


m_df = pd.merge(left = company_and_rounds_filt_clean,
    right = acquisition,
    how='left',
    left_on='company_ID',
    right_on='acquired_company_id',
    suffixes=('_company', '_acquisition'))
m_df.head()


# In[27]:


df = m_df[(m_df['funding_rounds'] > 0) | (m_df['investment_rounds'] > 0) | (m_df['status'].str.lower() == 'acquired')]
print(len(df))


# Данные объединены и отфильтрованы. Датасет соответствует предполагаемому размеру.

# In[28]:


dupl = df[['company_ID', 'name', 'status']].duplicated()
print(f"Количество дубликатов: {dupl.sum()}")
cdf = df.drop_duplicates(subset=['company_ID', 'name', 'status'])
print(f"Количество строк после удаления дубликатов: {len(cdf)}")


# 
# ### 3.2. Анализ выбросов
# 
# Заказчика интересует обычный для рассматриваемого периода размер средств, который предоставлялся компаниям.
# 
# - По предобработанному столбцу `funding_total` графическим способом оцените, какой размер общего финансирования для одной компании будет типичным, а какой — выбивающимся.
# - В процессе расчёта значений обратите внимание, например, на показатели, возвращаемые методом `.describe()`, — объясните их. Применимы ли к таким данным обычные способы нахождения типичных значений?

# In[29]:


plt.figure(figsize=(8, 4))
sns.boxplot(x=df['funding_total'])
plt.title('Распределение общего финансирования')
plt.xlabel('Общий размер финансирования')
plt.show()


# Выбросов много, они сильно искажают график. Ограничу выбросы 

# In[30]:


q75 = cdf['funding_total'].quantile(0.75)

# Фильтруем DataFrame — оставляем только значения <= 75-го квартиля
df_by_q75 = cdf[cdf['funding_total'] <= q75]

# Строим ящик с усами для отфильтрованных данных
plt.figure(figsize=(8, 4))
sns.boxplot(x=df_by_q75['funding_total'])
plt.title('Распределение общего финансирования (после фильтрации)')
plt.xlabel('Общий размер финансирования')
plt.show()


# In[31]:


des = cdf['funding_total'].describe()
desc = des.apply(lambda x: int(x))
print(desc)


# В данных очень большой разброс из-за экстремально высоких значений. Из-за этих же экстремальных значений принять как типичный размер финансирования среднее арифметическое будет не корректно. Наиболее объективно подойдет медианное значение равное 600 000. Это же мы видим при использовании метода describe и подтверждаем графиком.

# 
# ### 3.3. Куплены забесплатно?
# 
# - Исследуйте компании, которые были проданы за ноль или за один доллар, и при этом известно, что у них был ненулевой общий объём финансирования.
# 
# - Рассчитайте аналитически верхнюю и нижнюю границу выбросов для столбца `funding_total` и укажите, каким процентилям границы соответствуют.

# In[32]:


f_df = cdf[((cdf ['price_amount'] == 0) | (cdf['price_amount']==1)) & (cdf['funding_total']>0)]
f_df.info()


# In[33]:


Q1 = f_df['funding_total'].quantile(0.25)
Q3 = f_df['funding_total'].quantile(0.75)
print(Q1)
print(Q3)
IQR = Q3 - Q1
upper_bound = Q3 + 1.5 * IQR
lower_bound = Q1 - 1.5 * IQR
print(f'Верхний порог выбросов: {upper_bound}')
print(f'Нижний порог выбросов: {lower_bound}')


# Так как сумма покупки не может быть равно 0, именно 0 мы и берем как нижний порог выбросов.

# In[34]:


import numpy as np
values = f_df['funding_total'].values
sorted_values = np.sort(values)
position1 = np.searchsorted(sorted_values, upper_bound, side='left')
percentile1 = (position1 / len(sorted_values)) * 100
position2 = np.searchsorted(sorted_values, 0, side='left')
percentile2 = (position2 / len(sorted_values)) * 100

print(f"Значение верхнего порога принадлежит примерно к {percentile1:.2f}-му перцентилю")
print(f"Значение нижнего порога принадлежит примерно к {percentile2:.2f}-му перцентилю")


# 
# ### 3.4. Цены стартапов по категориям
# 
# Категории стартапов с типично высокими ценами покупки стартапов и значительным разбросом цен могут быть привлекательными для крупных инвесторов, которые готовы к высоким рискам ради потенциально больших доходов. Среди категорий стартапов выделите категории стартапов, характеризующиеся:
# 
# - типично высокими ценами;
# - и наибольшим разбросом цен за стартап.
# 
# Объясните, почему решили составить топ именно из такого числа категорий и почему рассчитывали именно так.

# In[43]:


median_by_category = cdf.groupby('category_code')['price_amount'].median().sort_values(ascending=False)
std_by_category = cdf.groupby('category_code')['price_amount'].std().sort_values(ascending=False)
print("Категории с типично высокими ценами (по медиане):")
for category, median_value in median_by_category.items():
    print(f"{category}: {round(median_value/1e6, 3)} млн")

print("\nКатегории с наибольшим разбросом цен (по стандартному отклонению):")
for category, std_value in std_by_category.items():
    print(f"{category}: {round(std_value/1e6,3)} млн")


# In[44]:


plt.figure(figsize=(14, 8))
sns.barplot(x=median_by_category/1e6, y=median_by_category.index)
plt.xlabel('Медиана стоимости (млн)')
plt.ylabel('Категория')
plt.title('Все категории по медиане стоимости стартапов')
plt.tight_layout()
plt.show()


# In[45]:


plt.figure(figsize=(12, 6))
sns.barplot(x=std_by_category.values/1e6, y=std_by_category.index)
plt.xlabel('Стандартное отклонение стоимости (млн)')
plt.ylabel('Категория')
plt.title('Категории по разбросу стоимости стартапов')
plt.show()


# In[46]:


top_hight_price = median_by_category.head(4)
top_std = std_by_category.head(7)
print("Топ-4 категорий с типично высокими ценами (по медиане):")
for category, median_value in top_hight_price.items():
    print(f"{category}: {round(median_value/1e6, 3)} млн")

print("\nТоп-7 категорий с наибольшим разбросом цен (по стандартному отклонению):")
for category, std_value in top_std.items():
    print(f"{category}: {round(std_value/1e6,3)} млн")


# In[47]:


plt.figure(figsize=(8, 4))
sns.barplot(x=top_hight_price/1e6, y=top_hight_price.index)
plt.xlabel('Медиана стоимости (млн)')
plt.ylabel('Категория')
plt.title('Топ-4 категорий с типично высокими ценами ')
plt.tight_layout()
plt.show()


# In[48]:


plt.figure(figsize=(8, 4))
sns.barplot(x=top_std/1e6, y=top_std.index)
plt.xlabel('Стандартное отклонение стоимости (млн)')
plt.ylabel('Категория')
plt.title('Топ-7 категорий с наибольшим разбросом цен')
plt.tight_layout()
plt.show()


# Из-за большого разброса в данных медиана более точно описывает среднюю сумму в сравнении с обычным средним арифметическим. На графиках по всем категориям четко виден разрыв с максимальными значениями, поэтому по медиане выбран топ-4, а по отклонению топ-7.

# 
# ### 3.5. Сколько раундов продержится стартап перед покупкой
# 
# - Необходимо проанализировать столбец `funding_rounds`. Исследуйте значения столбца. Заказчика интересует типичное значение количества раундов для каждого возможного статуса стартапа.
# - Постройте график, который отображает, сколько в среднем раундов финансирования проходило для стартапов из каждой группы. Сделайте выводы.

# In[49]:


fdf = cdf[cdf['funding_total']>0]
round_by_category = fdf.groupby('status')['funding_rounds'].mean().sort_values(ascending=False)
print(round_by_category)


# In[50]:


plt.figure(figsize=(8, 4))
sns.barplot(x=round_by_category.values, y=round_by_category.index, palette="coolwarm")
plt.xlabel('Среднее количество раундов финансирования')
plt.ylabel('Статус')
plt.title('Среднее количество раундов финансирования по статусам')
plt.tight_layout()
plt.show()


# Самое большое количество раундов финансирования у компаний, которые вышли на IPO.

# 
# ## Шаг 4. Итоговый вывод и рекомендации
# 
# Опишите, что было сделано в проекте, какие были сделаны выводы, подкрепляют ли они друг друга или заставляют сомневаться в полученных результатах.

# В ходе выполнения проекта были выполнены следующие шаги:
# 1. Загружены все необходимые библиотеки и датасеты.
# 2. Названия колонок и типы данных приведены к оптимальным вариантам.
# 3. Составлена сводная таблица и визуализация по годам финансирования. Видна тенденция увеличения раундов и снижение объема финансирования.
# 4. Получено представление об образовании сотрудников компаний путем объединения двух датасетов. Видна зависимость чем меньше компания по числу сотрудников, тем больший в них процент людей без образования.
# 5. Изучены выбросы.
# 6. Составлен Топ-5 категорий стартапов с высокой ценой и Топ-7 категорий с самым большим разбросом цены.
# 7. Определено среднее количество раундов финансирования компаний по их статусу.
