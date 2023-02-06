library(httr)
library(jsonlite)

response = GET("https://data.cityofchicago.org/resource/crimes.json") # getting crime data json
response # response will fetch JSON format data
