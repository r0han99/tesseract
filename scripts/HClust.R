library(plotly)
library(ggplot2)
library(ggdendro)
library(tidyr)


# Load required library
library(dplyr)

setwd("/Users/rohan/Desktop/Code/DataScience/Academic/Spring 23/CSCI 5622/Project/Chicago-Crime")


crimes <- read.csv('./Crime+Corpus/crime_cleaned_v3.csv')
head(crimes)


# Group by community area and primary type, count the number of occurrences
crime_count <- crimes %>%
  group_by(Community.Area.Name, Primary.Type) %>%
  summarize(count = n())

# Pivot the data to have primary type as columns
crime_count_pivot <- pivot_wider(crime_count, names_from = Primary.Type, values_from = count)

# Set row names to community area names
rownames(crime_count_pivot) <- crime_count_pivot$Community.Area.Name

# Remove the Community.Area column
#crime_count_pivot$Community.Area <- NULL

# View the resulting data frame
head(crime_count_pivot,2)



cosine_crime <- distance(as.matrix(crime_count_pivot), method="cosine"))
dist_cosine_crime<- as.dist(cosine_crime)

hc <- hclust(dist_cosine_crime, method = "ward.D2")
p <- ggdendrogram(hc, rotate = FALSE, size = 2) +   labs(title = "Community Areas Hirarchically Clustered based on the Frequency of Crimes Reported",x = "Community Areas", 
)

ggplotly(p) 

## DISTRICT

# Group by community area and primary type, count the number of occurrences
crime_count <- crimes %>%
  group_by(Dristrict.Name, Primary.Type) %>%
  summarize(count = n())

# Pivot the data to have primary type as columns
crime_count_pivot <- pivot_wider(crime_count, names_from = Primary.Type, values_from = count)

# Set row names to community area names
rownames(crime_count_pivot) <- crime_count_pivot$Dristrict.Name

# Remove the Community.Area column
#crime_count_pivot$Community.Area <- NULL

crime_count_pivot


cosine_crime <- distance(as.matrix(crime_count_pivot), method="cosine"))
dist_cosine_crime<- as.dist(cosine_crime)

hc <- hclust(dist_cosine_crime,method = "ward.D2")
p <- ggdendrogram(hc, rotate = FALSE, size = 2) + labs(title = "Districts Hirarchically Clustered based on the Frequency of Crimes Reported"
                                                       ,x = "Districts in Chicago",)


ggplotly(p)



# SOCIO ECONOMIC 
# Group by community area and primary type, count the number of occurrences
crime_count <- crimes %>%
  group_by(SocioEconomic.Status, Primary.Type) %>%
  summarize(count = n())

# Pivot the data to have primary type as columns
crime_count_pivot <- pivot_wider(crime_count, names_from = Primary.Type, values_from = count)

# Set row names to community area names
#rownames(crime_count_pivot) <- crime_count_pivot$Community.Area.Name

# Remove the Community.Area column
#crime_count_pivot$Community.Area <- NULL

# View the resulting data frame
df<- data.frame(crime_count_pivot)
df <- t(df[-1,])
df

# Set the first row as the column names
colnames(df) <- df[1,]

# Drop the original row
df <- df[-1,]
df

cosine_crime <- distance(as.matrix(df), method="cosine"))
dist_cosine_crime<- as.dist(cosine_crime)

hc <- hclust(dist_cosine_crime,method = "ward.D2")
p <- ggdendrogram(hc, rotate = FALSE, size = 2) + labs(title = "Crime Type Hirarchically Clustered on Socio-Economic Status ",x = "Community Areas", 
)

ggplotly(p)
