#############################################################
#                                                           #
#   R package Ridge Vs Python package frenezik : Ridge      #
#                                                           #  
#############################################################


# Work directory
setwd("C:/...")

# package installation
#install.packages('ridge')

# package import
library(ridge)
library(dplyr)
library(stargazer)
library(xtable)
library(ggplot2)


# california housing raw data
data.raw <- read.csv('housing.csv', nrows = 500)
data.raw <- data.raw[, -10] # drop ocean_proximity
stargazer(data.raw, type = "text")


# california housing scaled data
data.scaled <- read.csv('housing_scaled.csv')
stargazer(data.scaled, type = "text")


# Ridge package (1 = intercept / O = no intercept)
Ridge.scale <- linearRidge( median_house_value ~ . + 0, data = data.raw, lambda = 5, scaling = 'scale')
summary(Ridge.scale)

# Ridge package (1 = intercept / O = no intercept)
Ridge.none <- linearRidge( median_house_value ~ . + 0, data = data.scaled, lambda = 5, scaling = 'none')
summary(Ridge.none)

#############################################################
#                                                           #
# R ggplot Vs Python package frenezik : contour plot        #
#                                                           #  
#############################################################

bgrid <- expand.grid(
  b1 = seq(-10, 10, length.out = 50),
  b2 = seq(-10, 10, length.out = 50)
)

opt <- coef(lm(median_house_value ~. + 0, data = data.scaled))


y <- data.scaled$median_house_value
y <- data.matrix(y)
X <- data.scaled[, c('housing_median_age', 'latitude')] # drop median_house_value
X <- data.matrix(X)

rss <- function(b, x, y, b0) {
  yhat <- b0 + X %*% b
  sum((y - yhat)^2)
}

rss <- mutate(bgrid, rss = apply(bgrid, 1, rss, x = X, y = y, b0 = 0)) # opt[1]

ggplot(rss, aes(b1, b2, z = rss)) +
  geom_contour_filled() +
  geom_point(aes(x = opt[2], y = opt[3]), color = "red", size = 3) +
  labs(
    title = "Residual sum of squares",
    subtitle = "Contours of residual sum of squares",
    x = "Beta 2",
    y = "Beta 3"
  ) +
  theme_minimal() +
  theme(plot.background = element_rect(fill = "#fffbf2", colour = "transparent"))



#############################################################
#                                                           #
# R ggplot Vs Python package frenezik : surface plot        #
#                                                           #  
#############################################################


library(plotly)
library(dplyr)

y <- data.scaled$median_house_value
y <- data.matrix(y)
X <- data.scaled[, c('housing_median_age', 'latitude')] # drop median_house_value
X <- data.matrix(X)

bgrid <- expand.grid(
  b1 = seq(-10, 10, length.out = 50),
  b2 = seq(-10, 10, length.out = 50)
)

rss_func <- function(b, x_mat, y_vec, b0 = 0) {
  yhat <- b0 + x_mat %*% as.numeric(b)
  return(sum((y_vec - yhat)^2))
}

bgrid <- bgrid %>%
  rowwise() %>%
  mutate(rss_val = rss_func(c(b1, b2), X, y))

z_matrix <- matrix(bgrid$rss_val, nrow = 50, ncol = 50)

plot_ly(x = ~ unique(bgrid$b1), y = ~unique(bgrid$b2), z = ~z_matrix) %>% 
  add_surface()


