# 🚙 Car Prices Prediction 

[![](https://img.shields.io/badge/Python-FFD43B?style=for-the-badge&logo=python&logoColor=darkgreen)](https://www.python.org)  [![](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=TensorFlow&logoColor=white)](https://www.tensorflow.org) [![](https://img.shields.io/badge/scikit_learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/stable/) [![](https://img.shields.io/badge/SciPy-654FF0?style=for-the-badge&logo=SciPy&logoColor=white)](https://www.scipy.org) [![](https://img.shields.io/badge/Numpy-777BB4?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org) [![](https://img.shields.io/badge/Pandas-2C2D72?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org)  [![](https://img.shields.io/badge/Plotly-239120?style=for-the-badge&logo=plotly&logoColor=white)](https://plotly.com) [![](https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=Keras&logoColor=white)](https://keras.io) [![](https://img.shields.io/badge/conda-342B029.svg?&style=for-the-badge&logo=anaconda&logoColor=white)](https://www.anaconda.com)

## Introduction

The U.S. car manufacturing industry is thriving, constantly introducing vehicles equipped with advanced features and technology. These innovations, driven by a commitment to enhance the driving experience, ensure that motorists enjoy superior comfort and performance in their journeys.

<img src = "/images/cars.png" width = 350 height = 200/>

## Business Constraints / Key Performance Metrics (KPIs)

A key challenge in automotive sales is setting the optimal price for cars, a factor crucial in driving demand and sales volume. The pricing decision is influenced by multiple variables, including mileage, car size, and the manufacturer, among others. The complexity and multitude of these influencing factors make it a daunting task for human analysis to pinpoint the most effective pricing strategy. An effective solution to this challenge lies in harnessing machine learning and data science. These technologies enable the extraction of deep insights from vast datasets and facilitate accurate predictions. By applying these advanced analytical tools, companies can devise pricing strategies that not only resonate with market trends and consumer preferences but also bolster profitability.

## Machine Learning and Deep Learning

* <u>Machine Learning and Deep Learning Advancements:</u> Significant growth in machine learning and deep learning over the past decade, impacting various industries, including automotive.
* Predictive Pricing Models: Potential to predict car prices using key features like horsepower, make, and others, using machine learning techniques. 
* Optimal Pricing Strategies: Companies can utilize machine learning to determine the ideal price based on factors such as make, horsepower, and mileage, enhancing profitability. 
* Maximizing Profits Through Precise Pricing: Machine learning models enable setting the right price for new cars, ensuring maximum profit for manufacturers. 
* Car Prices Prediction Project: Focused on developing machine learning models to assist in accurate pricing of new cars, leading to cost savings for manufacturers.
* Data Analysis and Visualization: Initial steps include visualizing the data to understand critical information for accurate car price predictions.
* Utilizing Regression Techniques: Employing various regression methods to determine the average price of cars based on the dataset. 
* Comprehensive Pricing Insights: Aiming for a thorough understanding of pricing dynamics to aid in informed decision-making in the automotive sector.

<h2> Data Source</h2>

We will be analyzing a substantial dataset comprising approximately 10,000 data points. This dataset will be strategically split into training and test sets to ensure the efficacy of our predictive models. Our focus extends to various popular car models, integral to our daily lives, offering a practical perspective on their sales trends and average pricing structures. For a more in-depth understanding and to appreciate the scope of our analysis, you are encouraged to explore the dataset we utilized for car price prediction.

__Source:__ https://www.kaggle.com/CooperUnion/cardataset

## Metrics

Predicting car prices is an continuous machine learning problem, particularly in the realm of regression analysis. To effectively address this, we have employed several key metrics specifically tailored for regression problems. These metrics have been crucial in our process of predicting car prices, ensuring accuracy and reliability in our models. Here are the metrics we used:


* [__Mean Squared Error (MSE)__](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html)
* [__Mean Absolute Error (MAE)__](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_absolute_error.html)

## Exploratory Data Analysis (EDA)

In this section of the project, we conducted a thorough exploration of the data to uncover patterns, trends, and intriguing insights. Our analysis yielded several notable observations:

* Dominance of Certain Manufacturers: A significant portion of the cars in the dataset were from Chevrolet, with Ford being the second most common manufacturer.
* Peak Manufacturing Year: The year 2015 emerged as the year with the highest number of cars manufactured, according to the data.
* Data Incompleteness: The 'Market Category' feature had many missing values, along with some missing data in 'Engine HP' and 'Engine Cylinders'.
* Fluctuating Average Prices: The data revealed that the average car prices peaked in 2014 and were at their lowest in 1990.
* High-End Manufacturer Pricing: Cars from Bugatti were exceptionally expensive compared to other manufacturers.  
* Bugatti’s High Horsepower: Bugatti also stood out for its extremely high engine horsepower, as indicated by the graphical analysis in the notebook.
* Correlation Insights: There was a noticeable negative correlation between 'City Mileage' and features like 'Engine Cylinders' and 'Engine HP'. This suggests that cars with higher city mileage tend to have fewer cylinders and lower horsepower.

<h2> Visualizations</h2>

Upon examining the dataset, we observed several key categories including Vehicle Size, City MPG (miles per gallon), Popularity, and Transmission Types. Additionally, there are other features within the dataset that warrant further exploration through visual analysis. These visualizations will help in gaining a deeper understanding of the interplay between these diverse aspects and their impact on vehicle performance and consumer preferences.

<img src = "/images/data_head.png"/>

We will dive into a series of visualizations to gain insights into the dataset. By examining the distribution of car manufacturers represented, it becomes apparent that Chevrolet has the largest presence, with Ford following closely behind. These visual analyses are crucial in understanding the breadth and diversity of car brands within our data.

<img src = "/images/total_cars_by_manufacturer.png"/>

The visualizations clearly show a rising trend in car demand over the years, reflecting the growing automotive market. Additionally, this increase in recent data points suggests that our machine learning models are likely to be more accurate and effective when predicting prices for newer car models, due to the richer dataset available for these vehicles.

<img src = "/images/cars_manufactured_by_year.png"/>

The plot reveals that a majority of the cars in our dataset are automatic, outnumbering their manual counterparts. Additionally, there's a small segment of vehicles categorized as 'unknown', indicating some variability in transmission types beyond the standard automatic and manual classifications.

<img src = "/images/transmission_type.png"/>

The dataset predominantly features cars with regular unleaded engines, with a smaller yet notable presence of electric vehicles. Given their recent emergence, electric cars are less numerous. Additionally, a variety of other fuel types, including diesel, are represented in smaller quantities.

<img src = "/images/fuel_type.png"/>

A significant number of cars in the dataset are compact-sized, but there is also a fairly even distribution among different vehicle size categories.

<img src = "/images/vehicle_size.png"/>

Missingno plots highlight gaps in our dataset, particularly in 'Market Category' and 'Engine HP'. This helps us choose whether to impute or exclude these values to maintain our machine learning models' accuracy in price prediction.

<img src = "/images/missingno.png"/>

The analysis reveals that car prices peaked in 2014, marking it as the year with the highest average prices in our dataset. This is closely followed by 2012. Conversely, 1990 stands out as the year with the lowest average car prices. These trends highlight the year of manufacture as a significant factor in determining car prices, as evidenced by the plot below.

<img src = "/images/average_car_price_by_year.png"/>

The data shows that cars with both automatic and manual transmissions have the highest average prices, followed by automatic-only models. 'Unknown' transmission types in the data should be clarified or removed for more accurate analysis.

<img src = "/images/average_prices_by_transmission_type.png"/>

Luxury brands like Bugatti and Maybach top the average car prices, highlighting their market prestige and exclusivity. Other notable high-end brands include Rolls-Royce, Lamborghini, Bentley, McLaren, and Ferrari. The plot gives a clear view of the pricing trends among various car manufacturers.

<img src = "/images/average_msrp.png"/>

The bar chart showcases the horsepower across different car manufacturers, with Bugatti leading the pack. Despite its premium pricing, Bugatti stands out for its exceptional horsepower, followed by high-performance brands like McLaren and Maybach. These cars are renowned for their top-tier performance capabilities, as reflected in their impressive horsepower ratings.

<img src = "/images/engine_horsepower.png"/>

Popular car brands like Ford and BMW, along with Audi, Ferrari, Honda, and Nissan, lead in terms of consumer demand. Analyzing the influence of brand popularity on car pricing across these diverse manufacturers offers intriguing insights.

<img src = "/images/driven_wheel_configuration.png"/>

Segmenting the dataset by drivetrain reveals that 'all-wheel drive' (AWD) vehicles have the highest average prices, followed by 'rear-wheel drive' (RWD), with 'front-wheel drive' (FWD) being more affordable. This pricing reflects the performance capabilities and market preferences for each drivetrain type.

<img src = "/images/brand_popularity.png"/>

The left-hand plots display highway MPG for all dataset cars, revealing several outliers. Consequently, these outliers are removed, as seen in the right-hand plots. This refinement helps the ML model better learn and represent these features for accurate car price predictions.

<img src = "/images/highway_MPG.png"/><img src = "/images/highway_MPG_2.png"/>
<img src = "/images/city_MPG.png"/><img src = "/images/city_MPG_2.png"/>

The plot mirrors real-life trends, showing higher highway MPG compared to city MPG for most cars, evident in the differing spreads and means. While outliers exist, we'll retain some to gauge model performance.

<img src = "/images/city_MPG_and_highway_MPG.png"/>

The heatmap of the correlation matrix reveals a strong relationship between engine horsepower and engine cylinders, as well as a notable correlation between city mpg and highway mpg.

<img src = "/images/heatmap.png"/>

### Model Performance

Next, we'll evaluate the performance of different machine learning models on the test data using scatterplots. These visualizations will show the spread of our predictions compared to actual values.

[__Linear Regression__](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html) - For Linear Regression, the scatterplot indicates a strong performance, with predictions closely mirroring actual test outcomes. This model could be ideal for deployment scenarios where low latency is crucial.

<img src = "/images/lr.png"/>

[__Support Vector Regressor__](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html) - Support Vector Regression (SVR) is computationally intensive, and the results suggest that its predictions significantly deviate from actual car prices. Hence, exploring alternative models could be more effective.

<img src = "/images/svr.png"/>

[__K Nearest Regressor__](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html) - The K-Nearest Regressor demonstrates strong performance in predicting car prices, as evident in the plot. With minimal spread between the test outputs and the model's predictions, it's likely to yield low mean absolute error and mean squared error, indicating high accuracy.

<img src = "/images/k_nearest.png"/>

[__PLS Regression__](https://scikit-learn.org/stable/modules/generated/sklearn.cross_decomposition.PLSRegression.html) - While this model performs well in predicting car prices overall, it struggles to capture trends and patterns for higher-priced cars. This is evident from the significant spread among higher car price values, as depicted in the plot. In contrast, the K-Nearest Regressor excels in accurately predicting prices for higher-end vehicles.

<img src = "/images/pls.png"/>

[__Decision Tree Regressor__](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html) - Among all the tested models, the Decision Tree Regressor consistently outperformed the others. The plot illustrates a remarkable alignment between its predictions and the actual test values, showcasing its superior performance.

<img src = "/images/dt.png"/>

[__Gradient Boosting Regressor__](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html) - The performance plot of the Gradient Boosted Decision Regressor closely resembles that of the Decision Tree Regressor. However, at extremely high prices, the model struggles to capture the data trend. Nevertheless, it demonstrates solid overall performance.

<img src = "/images/gbr.png"/>

[__MLP Regressor__](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html) - While the MLP Regressor performs reasonably well in predicting car prices, it's worth noting that there are superior models available earlier in our analysis. These earlier models have demonstrated better performance, making them preferable choices in this context.

<img src = "/images/mlp.png"/>

[__Final Mean Absolute Error (MAE)__](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_absolute_error.html) - Following feature engineering and hyperparameter tuning, the Decision Tree Regressor emerged as the top-performing model, boasting the lowest mean absolute error (lower is better). In contrast, models like Support Vector Regressors were not only time-consuming to train but also delivered suboptimal results. Notably, Decision Tree Regressors not only excel in performance but are also highly interpretable, offering valuable insights into prediction rationale and feature importance in determining car prices.

<img src = "/images/mae.png"/>

[__Final Mean Squared Error (MSE)__](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html) - The Decision Tree Regressor achieved the highest performance when considering mean squared error as the output metric. While the Gradient Boosted Regressor came close in performance, the Decision Tree Regressor stands out due to its high interpretability and real-time deployability. As it consistently performs well across a wide range of metrics, choosing this model for deployment is a prudent choice. 

<img src = "/images/mse.png"/>

## Machine Learning Models 

We've employed a range of machine learning models to assess their impact on reducing mean absolute error (MAE) and mean squared error (MSE) on cross-validation data. The following list outlines the diverse set of models utilized for this evaluation.

| __Machine Learning Models__| __Mean Absolute Error__| __Mean Squared Error__|
| :-:| :-:| :-:|
| [__1. Linear Regression__](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html)| 6737| 364527989|
| [__2. Support Vector Regressor__](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html)|	22525|	2653742304|
| [__3. K Nearest Regressor__](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html)|	4668|	198923161|
| [__4. PLS Regression__](https://scikit-learn.org/stable/modules/generated/sklearn.cross_decomposition.PLSRegression.html)|	6732|	364661296|
| [__5. Decision Tree Regressor__](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html)|	__3327__|	__135789622__|
| [__6. Gradient Boosting Regressor__](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html)|	4432|	175275369|
| [__7. MLP Regressor__](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html)|	6467|	250908327|

## Outcomes

* The Decision Tree Regressor outperformed other models, achieving excellence in both mean absolute error (MAE) and mean squared error (MSE).
* Our Exploratory Data Analysis (EDA) identified Bugatti as a manufacturer with significantly higher prices than others in the dataset.
* Scatterplots depicting actual versus predicted prices demonstrated a nearly linear relationship, particularly evident with the Decision Tree Regressor model.

## Future Scope

* Integrating the top-performing machine learning model, the Decision Tree Regressor, into a live application would empower sellers to input details such as manufacturer, year of manufacture, and engine cylinders to optimize car prices for maximum profitability.
* Expanding the dataset with additional data points and features has the potential to further improve the model's capability to determine optimal prices for the latest cars in the market.

## Thank you!