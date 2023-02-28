library('devtools')
library('shiny')
library('arules')
library('arulesViz')


# Read in the Chicago Crime dataset
chicago_crime <- read.csv("./Crime+Corpus/crime_cleaned_v2.csv")

# Convert the Date column to a Date object
chicago_crime$Date <- as.Date(chicago_crime$Date, format = "%m/%d/%Y")


# Group the Primary Type column by Date, concatenating the crime types into a single string
grouped_data <- aggregate(Primary.Type ~ Date, chicago_crime, paste, collapse = ",")

# Convert the concatenated strings back to a list of crime types
grouped_data$Primary.Type <- strsplit(grouped_data$Primary.Type, ",")

# Convert the data to a binary matrix
transactions <- as(grouped_data$Primary.Type, "transactions")

# Inspect the transactions
inspect(transactions)


# APRIORI
rules <- apriori(transactions, parameter = list(supp=0.2, conf=0.9, 
                                                maxlen=2, 
                                                minlen=1,
                                                target= "rules"))


# checking if there are any rules with lift > 1.7
subset(rules, subset = lift > 1.7)


# Sorting rule by lift, Descending.
sort(rules, by='lift',decreasing=T)

# Scatterplot
plot(rules, method = "scatterplot")

# Graph Connection Plot with Limited Rules = 25
plot(rules, method = "graph", asEdges = TRUE, limit = 25) 

# Interactive Network Graph of Association Rules, Sorted by Lift in Descending Order
widget <- plot(head(sort(rules, by='lift',decreasing=T),n=50), method = "graph", measure = "lift", shading = "confidence", engine = "htmlwidget")
widget

# Saving Widget as HTML for embedding.
#saveWidget(widget, "Interactive_charts_html/chicago-crime-arules_top50-min2.html")


#==================================================
# COMMUNITY AREA WISE INSPECTION 
#==================================================

# AUSTIN, ILLINOIS
# Read in the Chicago Crime dataset
austin <- read.csv("./community_area/Austin.csv")

# Convert the Date column to a Date object
austin$Date <- as.Date(austin$Date)

# Group the Primary Type column by Date, concatenating the crime types into a single string
grouped_data <- aggregate(Primary.Type ~ Date, austin, paste, collapse = ",")

# Convert the concatenated strings back to a list of crime types
grouped_data$Primary.Type <- strsplit(grouped_data$Primary.Type, ",")

# Convert the data to a binary matrix
austin_transactions <- as(grouped_data$Primary.Type, "transactions")

# Inspect the transactions
inspect(austin_transactions)


# apriori on the above rules
austin_rules <- apriori(austin_transactions, 
                        parameter = list(supp=0.3, conf=0.5, 
                                         minlen=2,
                                         maxlen=4,
                                         target= "rules"))
# Sorting rule by lift, Descending.
inspect(sort(austin_rules, by="lift", decreasing=TRUE))

# Making Subset of the rules, top50 by lift
subrules <- head(austin_rules, n = 50, by = "lift")
inspect(head(subrules))

# Static Graph
plot(subrules, method = "graph", measure = "lift", shading = "confidence")

# Interactive Graph
widget <- plot(subrules, method = "graph", measure = "lift", shading = "confidence", engine = "htmlwidget")
widget

# Saving HTML widget
#saveWidget(widget, "Interactive_charts_html/austin_rules.html")


# NEAR NORTH SIDE, ILLINOIS

# Read in the Chicago Crime dataset
northside <- read.csv("./community_area/Near North Side.csv")

# Convert the Date column to a Date object
northside$Date <- as.Date(northside$Date)

# Group the Primary Type column by Date, concatenating the crime types into a single string
grouped_data <- aggregate(Primary.Type ~ Date, northside, paste, collapse = ",")

# Convert the concatenated strings back to a list of crime types
grouped_data$Primary.Type <- strsplit(grouped_data$Primary.Type, ",")

# Convert the data to a binary matrix
northside_transactions <- as(grouped_data$Primary.Type, "transactions")

# Inspect the transactions
inspect(northside_transactions)

# Apriori 
northside_rules <- apriori(northside_transactions, 
                           parameter = list(supp=0.3, conf=0.7, 
                                            minlen=1,
                                            maxlen=3,
                                            target= "rules"))

# subset of rules 
subrules <- head(northside_rules, n = 50, by = "lift")


# Static Graph
plot(subrules, method = "graph", measure = "lift", shading = "confidence")

widget <- plot(subrules, method = "graph", measure = "lift", shading = "confidence", engine = "html")
widget

# Saving HTMLwidget
#saveWidget(widget, "Interactive_charts_html/near_north.html")

# Connection Graph 
plot(subrules, method = "graph", asEdges = TRUE, limit = 10) 


inspect(head(subrules))