Домашенее Задание по курсу "ML".

Подготовил Тимур Ермешев

____

**Название проекта:**

Создание и обучение линейной модели регрессии и запуск ее работы с использованием FastAPI

____


**Постановка задачи:** В этом домашнем задании вам предлагается обучить модель регрессии для предсказания стоимости автомобилей, а также реализовать веб-сервис для применения построенной модели на новых данных.

____


**Описание задачи.**
Провести предобработку данных.
Провести анализ.
Обучить линейные модели регрессии.
Сделать подбор гиперпараметров при помощи GridSearchCV

____


**Исходные данные:**

Данные состоят из двух датасетов с параметрами автомобилей и их ценами:

 * [Тестовый]('https://raw.githubusercontent.com/hse-mlds/ml/main/hometasks/HT1/cars_test.csv') 
 * [Обучающий]('https://raw.githubusercontent.com/hse-mlds/ml/main/hometasks/HT1/cars_train.csv')

____


**Структура репозитория:**

**HW_01_ML_ERMESHEV_TIMUR.ipynb** - тетрадка анализом и подбором модели

**Pipline_train.py** - Финальный скрипт для обучения модели

**/models/LinearRegression_model.pkl** - сохраненная предобученная модель LinearRegression

**/models/ohe_model.pkl** - сохраненная предобученная модель OHE

**/models/median_values_model.data** - сохраненные значения средних значений для заполнения пустот

**main.py** - Финальный скрипт для предсказаний с использованием FastAPI

**/sample_for_test/test_file.csv** - пример входного файла для тестирования сервиса

**prediction.csv** - Выходной файл с предсказаниями

**/pic/** - директория со скриншотами работы fastapi

____


**Порядок работы:**

1.	Загружены два датасета для обучения и тестирования по ссылке и загружены в датафрейм.
2.	Проверены типы данных командой df.info()
3.	Проверено наличие пропусков командой df.isna().sum()
4.	Посчитаны основные метрики командой df.describe()
5.	Проверено количество полных дубликатов командой df.duplicated().sum(), отличающихся только по целевому признаку. 
6.	Дубликаты удалены с оставлением одного из них df.drop_duplicates(keep='first'). 
7.	Индексы сброшены командой df.reset_index(drop=True)
8.	Столбец “mileage” имеет разные единицы измерения: kmpl и km/kg.
9.	Написана функция, которая переводит единицы km/kg к kmpl, так как их меньше. При этом используется коэффициент перевода, учитывающий плотность бензина. По идее нужно было еще проверить есть ли среди этих авто, которые работают на других видах топлива, но из-за небольшого количества позиций, можно взять одну среднюю единицу. 
10.	Столбцы ‘mileage', 'engine', 'max_power' имеют единицы измерения в значениях. Для приведения их в числовой вид написана функция. 
11.	Столбец 'torque' имеет в себе значения двух параметров которые обозначают максимальную мощность при максимальном крутящем моменте. Написана функция, которая разбивает значение ячейки на 2 столбца 'torque', 'max_torque_rpm' и записывает в них соответствующее значение в числовом виде, откидывая при этом единицы измерения. Также максимальная мощность имеет разные единицы измерения, что тоже учитывается путем умножения на соответствующий коэффициент
12.	Пропуски заполнены медианным значением по столбцу. 
13.	Количество посадочных мест представлено в числовом виде, хотя, по сути, этот признак определяет категорию автомобиля. Поэтому в финальном Пайплайн этот признак переведен в категориальный.
14.	Для всех признаков построен график pairplot() библиотеки seaborn. Данный график представляет собой набор графиков зависимости друг от друга всех признаков в датасете (каждый с каждым). На графиках видна слабо выраженная зависимость целевого признака от year, engine, torque и выраженная зависимость от признака max_power. Также можно сказать, что по этим признакам с целевым признаком есть положительная корреляция.
15.	Произведен расчет корреляции числовых признаков командой df.corr() и для результатов построен график heatmap() библиотеки seaborn, который цветом и числами показывает корреляцию признаков (каждый с каждым). На карте видно, что year и engine имеют наименьшую линейную зависимость между собой. А engine и max_power имеют наибольшую линейную зависимость между собой, если не считать линейную зависимость max_power и целевого признака.
16.	Построен график зависимости максимальной мощности от объема двигателя, для которых видна выраженная корреляция. 
17.	Построен график зависимости километража от года выпуска. График показывает, что старые автомобили не имеют более высокий километраж. Скорей всего старые автомобили потому и на ходу до сих пор, что они используются не так активно.
18.	Построен график распределения цены автомобилей и добавлено нормальное распределение. На графике видно, что целевой признак имеет не нормальное распределение со смещением влево и большим хвостом справа. Скорей всего линейные модели будут иметь низкую точность и возможно придется модифицировать целевой признак.
19.	Произведено логарифмирование целевого признака и построен такой же график распределения логарифмированной цены автомобилей. После логарифмирования получаем более приближенное распределение к нормальному.
20.	Произведено обучение на дефолтных параметрах моделями LinearRegression, Lasso и с подбором гиперпараметров для моделей Lasso, ElasticNet, Ridge при помощи GridSearchCV. Обучение проводилось на датасете, у которого все категориальные признаки ('name', 'fuel', 'seller_type', 'transmission', 'owner') удалены, пропуски заполнены медианным значением по столбцу, целевой признак не преобразован. В результате получена метрика R2 в районе 0.60 +- 0.02. манипуляции с добавление стандартизации и преобразования категориальных признаков ('fuel', 'seller_type', 'transmission', 'owner') методом OHE существенных результатов не принесло.
21.	Написана функция бизнесовой метрики.

Для финальной реализации написан пайплайн, который обучает модель и сохраняет предобученные модели.

1.	Добавлен столбец с отношением числа "лошадей" на литр объема
2.	Добавлен столбец с квадратом года выпуска
3.	Реализована функция, для вытаскивания из столбца ‘name’ название бренда и модели автомобиля и записывается в соответствующие столбцы ‘brand’ и ’model’.
4.	Реализована функция по заполнению пропусков медианным значением. При этом медианное значение применяется для соответствующего значения бренда и модели. Таким образом заполнена большая часть пропусков в обучающей выборке и все пропуске в тестовой выборке. Медианные значения записаны в файл для дальнейшего использования.
5.	Оставшиеся пропуски заполнены медианным значением по столбцу. Можно было бы удалить данные строки, так как они составляют менее 5 % от всей выборки.
6.	Произведено логарифмирование целевого признака.
7.	Произведено OHE кодирование категориальных признаков и модель кодирования записана в файл. 

8.	Результатом предобрабоки и добавления новых признаков (Feature Engineering) удалось добиться R2 метрики = 0.923 и бизнес метрики 0.354 на дефолтных параметрах для LinearRegression. Остальные линейные модели показали немного меньший результат.
9.	При оставлении и кодировании столбца с названием модели авто, R2 увеличивается до 0.955, но при этом становится большое количество столбцов (около 5000)
10.	Добавление бренда, квадрата года выпуска, отношение числа "лошадей" на литр объема дало небольшой прирост в метрике. Логарифмирование целевого признака дало существенный прирост. 
11.	Итоговая модель отбрасывает столбец с названием модели авто и при этом получается R2 метрика = 0.923 и бизнес метрика 0.354 на дефолтных параметрах для LinearRegression.
