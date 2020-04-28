from datetime import datetime, timedelta
from dateutil import tz
from IPython.display import display, Markdown, Latex
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pylab import rcParams
from Regression import RidgeRegression
import warnings

rcParams['figure.figsize'] = 9, 5

deaths_url = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv"
confirmed_cases_url = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv"

confirmed_cases = pd.read_csv(confirmed_cases_url)
deaths = pd.read_csv(deaths_url)

world_pop = pd.read_csv("World Population.csv")

deaths_clean = deaths.groupby("Country/Region").sum().drop(["Lat", "Long"], axis=1)
confirmed_cases_clean = confirmed_cases.groupby("Country/Region").sum().drop(["Lat", "Long"], axis=1)
confirmed_cases_clean.head()

world_pop_clean = world_pop.loc[:, [world_pop.columns[0], "2020"]].copy()#world_pop[[world_pop.columns[0], "2020"]]
world_pop_clean["2020"] *= 1000
world_pop_clean = world_pop_clean.set_index("Region, subregion, country or area *")
world_pop_clean.columns = ["2020 Population"]
world_pop_clean.head()

coronaToPopulation = {
    "Bolivia" : "Bolivia (Plurinational State of)",
    "Brunei" : "Brunei Darussalam",
    "Congo (Brazzaville)" : None,
    "Congo (Kinshasa)" : None,
    "Cote d'Ivoire" : None,
    "Diamond Princess" : None,
    "Iran" : "Iran (Islamic Republic of)",
    "Korea, South" : "Republic of Korea",
    "Laos" : None,
    "Moldova" : "Republic of Moldova",
    "Russia" : "Russian Federation",
    "Syria" : "Syrian Arab Republic",
    "Taiwan*" : "China, Taiwan Province of China",
    "Tanzania" : "United Republic of Tanzania",
    "West Bank and Gaza" : None,
    "US" : "United States of America",
    "Venezuela" : "Venezuela (Bolivarian Republic of)",
    "Vietnam" : "Viet Nam",
    "Kosovo" : None,
    "Burma" : None
}

missingCountries = list(confirmed_cases_clean.index[[c not in world_pop_clean.index for c in confirmed_cases_clean.index]])

for m in missingCountries:
    if m not in coronaToPopulation.keys():
        coronaToPopulation[m] = None

assert np.all(deaths_clean.index == confirmed_cases_clean.index), "Death data and confirmed cases data have different indices!"
coronaIndex = list(confirmed_cases_clean.index)
for k, v in coronaToPopulation.items():
    try:
        coronaIndex[coronaIndex.index(k)] = v
    except:
        pass

confirmed_cases_clean.index = coronaIndex
deaths_clean.index = coronaIndex

confirmed_cases_clean = confirmed_cases_clean.loc[pd.isna(confirmed_cases_clean.index) == False, :]
deaths_clean = deaths_clean.loc[pd.isna(deaths_clean.index) == False, :]

confirmed_cases_clean = confirmed_cases_clean[np.sum(confirmed_cases_clean, axis=1) > 1]
deaths_clean = deaths_clean[np.sum(deaths_clean, axis=1) > 1]

confirmed_cases_clean.head()


confirmed_cases_clean.loc["China, Taiwan Province of China"]

assert np.all([c in world_pop_clean.index for c in confirmed_cases_clean.index]), "A country in confirmed_cases_clean is not in the world_pop_clean! Investigate further."
assert np.all([c in world_pop_clean.index for c in deaths_clean.index]), "A country in deaths_clean is not in the world_pop_clean! Investigate further."
print("All Coronavirus countries properly match with the World Population data! Safe to proceed.")

def normalize(data, populations, country):
    assert country in populations.index, f"{country} is not in the World Populations dataset!"
    
    country_population = populations.loc[country][0]
    data = data / country_population
    return data

def reverseNormalize(data, populations, country):
    assert country in populations.index, f"{country} is not in the World Populations dataset!"
    
    country_population = populations.loc[country][0]
    data = data * country_population
    return data

def countryToData(baseTable, country="Italy", threshold=1, windowSize=4, shouldNormalize=True):
    assert country in list(baseTable.index), f"Invalid Country: {country}"
    windowSize += 1
    row = baseTable.loc[country]
    if not np.any(row >= threshold):
        return None
    y = np.array(row[np.argmax(row >= threshold):])
    if len(y) <= windowSize:
        return None
    X = []
    for i in range(len(y) - windowSize):
        window = y[i:i+windowSize]
        X.append(window)
    if shouldNormalize:
        return normalize(np.array(X), world_pop_clean, country)
    return np.array(X)

def generateData(baseTable, trainCountries=[], valCountries=["Italy"], threshold=1, windowSize=4, exclude=[]): #To train the model we use Italy
    if len(trainCountries) == 0:
        trainCountries = list(set(baseTable.index) - set(valCountries))
    train = None
    val = None
    
    for country in trainCountries + valCountries:
        countryData = countryToData(baseTable, country, threshold=threshold, windowSize=windowSize)
        if countryData is None or country in exclude:
            continue
        if country in trainCountries:
            if train is None:
                train = countryData
            else: 
                train = np.vstack((train, countryData))
        if country in valCountries:
            if val is None:
                val = countryData
            else:
                val = np.vstack((val, countryData))
    return train, val

def logSafe(data, base=np.e):
    return np.log(np.clip(data, 1e-15, 1)) / np.log(base)

def expSafe(data, base=np.e):
    return base ** data

countryToPredict = "Taiwan"

train_data_cases, val_data_cases = generateData(confirmed_cases_clean, 
                          trainCountries=[], 
                          valCountries=[countryToPredict], 
                          threshold=1, 
                          windowSize=4)

model = RidgeRegression(inputFunction = logSafe, 
                        outputFunction = lambda data: reverseNormalize(expSafe(data), 
                                                                       world_pop_clean, 
                                                                       countryToPredict))

model.train(train_data_cases)
todayPred = model.predict(val_data_cases[-1,1:])[0]
print(f"This example model predicted that today, The United States would have {int(todayPred)} total cases of Coronavirus.")

def loss(model, val_data, country="United States of America", metric="mae", n=3):
    prediction = model.predict(val_data[:, :-1])[-n]
    truth = val_data[-n:,-1]
    truth = reverseNormalize(truth, world_pop_clean, country)
    if metric == "mse":
        return np.mean((truth - prediction)**2)
    elif metric == "mae":
        return np.mean(np.abs(truth - prediction))

loss(model, val_data_cases)

def scoreParameters(baseTable=confirmed_cases_clean, country="United States of America", 
                    threshold=1, windowSize=15, lmb=1e-5, metric="mae", n=3, logBase=np.e):
    train_data, val_data = generateData(baseTable, 
                              trainCountries=[], 
                              valCountries=[country], 
                              threshold=threshold, 
                              windowSize=windowSize)
#     print(train_data)
    
    model = RidgeRegression(inputFunction = lambda data: logSafe(data, logBase), 
                            outputFunction = lambda data: reverseNormalize(expSafe(data, logBase), 
                                                                           world_pop_clean, 
                                                                           country))
    model.train(train_data, lmb=lmb)
    return loss(model, val_data, country, metric, n)


def visualize(values_tested, losses, param, xLog=False):
    plt.plot(values_tested, losses, label="Loss")
    plt.xlabel(f"Values of {param}")
    plt.ylabel(f"Loss")
    if xLog: plt.xscale("log")
    plt.title(f"Testing Different Values of {param}")
    plt.show()


thresholds_to_test = list(np.linspace(0, 100, 30))
threshold_losses = []
for t in thresholds_to_test:
    threshold_loss = scoreParameters(threshold=t, lmb=0)
    threshold_losses.append(threshold_loss)

best_threshold_i = np.argmin(threshold_losses)
best_threshold_cases = thresholds_to_test[best_threshold_i]
visualize(thresholds_to_test, threshold_losses, "threshold")

print(f"The best threshold found was {best_threshold_cases} with a loss of {threshold_losses[best_threshold_i]}")

windowSizes_to_test = list(range(10,29))
windowSize_losses = []
for w in windowSizes_to_test:
    windowSize_loss = scoreParameters(threshold=best_threshold_cases, windowSize=w, lmb=0)
    windowSize_losses.append(windowSize_loss)

best_windowSize_i = np.argmin(windowSize_losses)
best_windowSize_cases = windowSizes_to_test[best_windowSize_i]
visualize(windowSizes_to_test, windowSize_losses, "windowSize")

print(f"The best windowSize found was {best_windowSize_cases} with a loss of {windowSize_losses[best_windowSize_i]}")

lmb_to_test = list(np.geomspace(1e-10, 10, 30))
lmb_losses = []
for l in lmb_to_test:
    lmb_loss = scoreParameters(threshold=best_threshold_cases, windowSize=best_windowSize_cases, lmb=l)
    lmb_losses.append(lmb_loss)

best_lmb_i = np.argmin(lmb_losses)
best_lmb_cases = lmb_to_test[best_lmb_i]
visualize(lmb_to_test, lmb_losses, "lambda", True)

print(f"The best lambda found was {best_lmb_cases} with a loss of {lmb_losses[best_lmb_i]}")

thresholds_to_test = list(range(1,12))
threshold_losses = []
for t in thresholds_to_test:
    threshold_loss = scoreParameters(deaths_clean, threshold=t, lmb=0)
    threshold_losses.append(threshold_loss)

best_threshold_i = np.argmin(threshold_losses)
best_threshold_deaths = thresholds_to_test[best_threshold_i]
visualize(thresholds_to_test, threshold_losses, "threshold")

print(f"The best threshold found was {best_threshold_deaths} with a loss of {threshold_losses[best_threshold_i]}")

windowSizes_to_test = list(range(1,15))
windowSize_losses = []
for w in windowSizes_to_test:
    windowSize_loss = scoreParameters(deaths_clean, threshold=1, windowSize=w, lmb=0.1)
    windowSize_losses.append(windowSize_loss)

best_windowSize_i = np.argmin(windowSize_losses)
best_windowSize_deaths = windowSizes_to_test[best_windowSize_i]
visualize(windowSizes_to_test, windowSize_losses, "windowSize")

print(f"The best windowSize found was {best_windowSize_deaths} with a loss of {windowSize_losses[best_windowSize_i]}")

lmb_to_test = list(np.geomspace(1e-15, 1e-5, 30))
lmb_losses = []
for l in lmb_to_test:
    lmb_loss = scoreParameters(deaths_clean, threshold=best_threshold_deaths, windowSize=best_windowSize_deaths, lmb=l)
    lmb_losses.append(lmb_loss)

best_lmb_i = np.argmin(lmb_losses)
best_lmb_deaths = lmb_to_test[best_lmb_i]
visualize(lmb_to_test, lmb_losses, "lambda", True)

print(f"The best lambda found was {best_lmb_deaths} with a loss of {lmb_losses[best_lmb_i]}")

def predict(model, baseTable=confirmed_cases_clean, windowSize=5, daysToPredict=5, country="Taiwan"):
    pop = world_pop_clean.loc[country][0]
    data = baseTable.loc[country, :][-windowSize:] / pop
    predictions = []
    while len(predictions) < daysToPredict:
        nextData = data[-windowSize:]
        nextPred = model.predict(nextData)[0]
        predictions.append(nextPred)
        data = np.append(data, normalize(nextPred, world_pop_clean, country))
    return predictions

def timeFormatter(daysInFuture=1):
    latestFromData = datetime.strptime(confirmed_cases_clean.columns[-1], "%m/%d/%y")
    latestFromData = latestFromData.replace(tzinfo=tz.gettz('UTC'))
    locTime = latestFromData.astimezone(tz.tzlocal()) + timedelta(days=daysInFuture + 1)
    return locTime.strftime("%b %d, %Y, %l:%M%p")

def printPredictions(preds, suffix="total cases of Coronavirus in The United States."):
    for i in range(len(preds)):
        p = preds[i]
        i += 1
        s = f"On {timeFormatter(i)}: I predict {int(p):,} {suffix}\n"
        print(s)


countryToPredict = "Taiwan"

train_data_cases, val_data_cases = generateData(confirmed_cases_clean, 
                          trainCountries=[], 
                          valCountries=[countryToPredict], 
                          threshold=best_threshold_cases, 
                          windowSize=best_windowSize_cases)


total_cases_model = RidgeRegression(inputFunction = logSafe, 
                                    outputFunction = lambda data: reverseNormalize(expSafe(data), 
                                                                                   world_pop_clean, 
                                                                                   countryToPredict))
total_cases_model.train(train_data_cases, lmb=best_lmb_cases)

total_cases_predictions = predict(total_cases_model, confirmed_cases_clean, best_windowSize_cases, daysToPredict=5)

printPredictions(total_cases_predictions)

def latexifyExponent(base="e",exp="exponent"):
    return f"{base}^{{{exp}}}"

def buildLatexEquation(weights, decimals=4):
    s = ""
    bias = weights[0]
    s += f"P^{{{round(-sum(weights[1:]), decimals)}}}*{latexifyExponent('e', round(bias, decimals))}"
    weights = weights[1:]
    weight_sum = sum(weights[1:])
    for i in range(len(weights)):
        i = len(weights) - i - 1
        s += latexifyExponent(f"*X_{{{len(weights) - i}}}", round(weights[i], decimals))
    return s


display(Latex("$X_{{Tomorrow}} = $"))
display(Latex(f"${buildLatexEquation(total_cases_model.w)}$"))

daysToPlot = 10
plt.plot(list(range(daysToPlot)), predict(total_cases_model, confirmed_cases_clean, best_windowSize_cases, daysToPredict=daysToPlot))
plt.xlabel(f"Days after {timeFormatter(0)}")
plt.ylabel("Total Number of Cases In Taiwan (Predicted)")
plt.show()

daysToPlot = 2000
plt.plot(list(range(daysToPlot)), predict(total_cases_model, confirmed_cases_clean, best_windowSize_cases, daysToPredict=daysToPlot))
plt.xlabel(f"Days after {timeFormatter(0)}")
plt.ylabel("Total Number of Predicted Cases In Taiwan (10-Millions)")
plt.show()

countryToPredict = "Taiwan"

train_data_deaths, val_data_deaths = generateData(deaths_clean, 
                          trainCountries=[], 
                          valCountries=[countryToPredict], 
                          threshold=best_threshold_deaths, 
                          windowSize=best_windowSize_deaths)

total_deaths_model = RidgeRegression(inputFunction = logSafe, 
                                    outputFunction = lambda data: reverseNormalize(expSafe(data), 
                                                                                   world_pop_clean, 
                                                                                   countryToPredict))
total_deaths_model.train(train_data_deaths, lmb=best_lmb_deaths)

death_preds = predict(total_deaths_model, deaths_clean, best_windowSize_deaths, daysToPredict=5)

printPredictions(death_preds, suffix="total deaths from Coronavirus in The United States.")