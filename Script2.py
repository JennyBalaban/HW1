import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# The data source can be found below, It was licenced by MIT
# https://www.kaggle.com/datasets/shadesh/bike-seles-dataset

data = pd.read_csv('/Users/jenny/Desktop/sales_data.csv')
print(data.head())
print(data.shape)

unique = data['Country'].unique()
print(unique)

unique_products = data['Product'].unique()
print(unique_products)

products_to_drop = ['All-Purpose Bike Stand', 'Mountain Bottle Cage', 'Water Bottle - 30 oz.', 'Road Bottle Cage','AWC Logo Cap',
                    'Bike Wash - Dissolver', 'Fender Set - Mountain', 'Half-Finger Gloves, L', 'Half-Finger Gloves, M', 
                    'Half-Finger Gloves, S', 'Sport-100 Helmet, Black', 'Sport-100 Helmet, Red', 'Sport-100 Helmet, Blue',
                    'Hydration Pack - 70 oz.', 'Short-Sleeve Classic Jersey, XL', 'Short-Sleeve Classic Jersey, L', 
                    'Short-Sleeve Classic Jersey, M', 'Short-Sleeve Classic Jersey, S', 'Long-Sleeve Logo Jersey, M', 
                    'Long-Sleeve Logo Jersey, XL', 'Long-Sleeve Logo Jersey, L', 'Long-Sleeve Logo Jersey, S', 'Mountain-100 Silver, 38', 
                    "Women's Mountain Shorts, M", "Women's Mountain Shorts, S", "Women's Mountain Shorts, L", 'Racing Socks, L',
                    'Racing Socks, M', 'Mountain Tire Tube', 'Touring Tire Tube', 'Patch Kit/8 Patches', 'HL Mountain Tire', 'LL Mountain Tire'
                    'Road Tire Tube', 'LL Road Tire', 'Touring Tire', 'ML Mountain Tire', 'HL Road Tire', 'ML Road Tire', 'Classic Vest, L', 
                    'Classic Vest, M', 'Classic Vest, S', 'LL Mountain Tire', 'Road Tire Tube'] 


data = data[~data['Product'].isin(products_to_drop)]


data_filtered = data.filter(items=['Date', 'Order_Quantity', 'Revenue', "Country"])
print(data_filtered.head())

print(data_filtered.isnull().sum())

summary_stats = data_filtered[['Order_Quantity', 'Revenue']].describe()
print(summary_stats)

data_filtered = data_filtered.sort_values(by='Date')
data_filtered.reset_index(drop=True, inplace=True)
print(data_filtered.head())

data_filtered['Date'] = pd.to_datetime(data_filtered['Date'], errors='coerce')
data_filtered['Year-Month'] = data_filtered['Date'].dt.to_period('M')

monthly_sales = data_filtered.groupby('Year-Month')['Order_Quantity'].sum().reset_index()
print(monthly_sales.head())

cumulative_sales = monthly_sales['Order_Quantity'].cumsum()

print(cumulative_sales.head())
cumulative_sales.shape

country_counts = data_filtered['Country'].value_counts()  
country_percentage = (country_counts / country_counts.sum()) * 100  

print(country_percentage)

#%pip install scipy
from scipy.optimize import curve_fit

def bass_model(t, p, q, m):
    """
    t: time (months or periods)
    p: coefficient of innovation
    q: coefficient of imitation
    m: market potential (maximum number of adopters)
    """
    adoption = (p + q) ** 2 / p * (np.exp(-(p + q) * t) / ((1 + q/p * np.exp(-(p + q) * t)) ** 2))
    return m * adoption

time_periods = np.arange(1, len(cumulative_sales)+1)

popt, _ = curve_fit(bass_model, time_periods, cumulative_sales, bounds=(0, [1, 1, 1000000]))

p, q, m = popt
print(f"Estimated parameters:\np: {p}\nq: {q}\nm: {m}")


future_time_periods = np.arange(len(cumulative_sales) + 1, len(cumulative_sales) + 13)
future_sales = bass_model(future_time_periods, p, q, m)


print(f"Predicted future sales for the next 12 months:\n{future_sales}")

# Plotting
plt.figure(figsize=(10, 5))
plt.plot(np.arange(1, len(cumulative_sales) + 1), cumulative_sales, label='Actual Cumulative Sales', marker='o')

# Ensure cumulative_sales is not empty before accessing its last element
if not cumulative_sales.empty:
    plt.plot(future_time_periods, future_sales.cumsum() + cumulative_sales.iloc[-1], label='Predicted Future Sales', marker='x', linestyle='--')
else:
    print("Cumulative sales data is empty; cannot plot future sales.")

plt.xlabel('Time Periods (Months)')
plt.ylabel('Cumulative Sales')
plt.title('Cumulative Sales and Future Predictions')
plt.legend()
plt.show()

p = 0.00001  
q = 0.4    
M = 1000000     


T = np.arange(1, 60)  # 5 years

def bass_diffusion(p, q, M, T):
    adoption = np.zeros(len(T))
    cumulative_adoption = np.zeros(len(T))
    
    for t in range(1, len(T)):
        F_t = 1 - np.exp(-(p + q) * T[t])
        cumulative_adoption[t] = M * F_t
        adoption[t] = cumulative_adoption[t] - cumulative_adoption[t - 1]
    
    return adoption, cumulative_adoption

adopters_per_period, cumulative_adopters = bass_diffusion(p, q, M, T)

print("Adopters per period (first 12 months):")
print(adopters_per_period[:12])

print("\nCumulative adopters (first 12 months):")
print(cumulative_adopters[:12])

# Plot the results
plt.figure(figsize=(10, 5))
plt.plot(T, cumulative_adopters, 'o-', label='Cumulative Adopters')
plt.plot(T, adopters_per_period, 'x--', label='Adopters per Period')
plt.xlabel('Time Periods (Months)')
plt.ylabel('Number of Adopters')
plt.title('Bass Diffusion Model: Adopters and Cumulative Adopters')
plt.legend()
plt.grid(True)
plt.show()
