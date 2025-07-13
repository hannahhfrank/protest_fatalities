setwd('/Users/hannahfrank/protest_fatalities')
library(lmtest)
library(sandwich)
library(Hmisc)
library(plm)
library(AER)
library(lavaan)
library(semPlot)
library(ggplot2)
library(stargazer)
library(lmtest)
library(car)
library(tidyverse)

# Load data
df <- read.csv('data/final_shapes_s.csv')
df$cluster_1<-as.factor(df$cluster_1)
df$cluster_2<-as.factor(df$cluster_2)
df$cluster_3<-as.factor(df$cluster_3)
df$cluster_4<-as.factor(df$cluster_4)
df$cluster_5<-as.factor(df$cluster_5)
df$clusters_cen<-as.factor(df$clusters_cen)
df$clusters_cen<-relevel(df$clusters_cen,ref="4")
levels(df$clusters_cen)
#df[is.na(df)] <- 0

# Remove missing values
MISSING <- is.na(df$n_protest_events_norm_lag_1 ) |
  is.na(df$n_protest_events_norm_lag_2) |
  is.na(df$n_protest_events_norm_lag_3) |
  is.na(df$fatalities_log_lag1) 
df_s <- subset(df, subset = !MISSING)

# Linear regression models -------------

lm1 <- lm(fatalities_log ~ n_protest_events_norm + n_protest_events_norm_lag_1 + n_protest_events_norm_lag_2 + n_protest_events_norm_lag_3 + fatalities_log_lag1+NY.GDP.PCAP.CD_log+SP.POP.TOTL_log+v2x_libdem+v2x_clphy+v2x_corr+v2x_rule+v2x_civlib+v2x_neopat, data = df_s)
summary(lm1)

lm2 <- lm(fatalities_log ~ n_protest_events_norm + n_protest_events_norm_lag_1 + n_protest_events_norm_lag_2 + n_protest_events_norm_lag_3 + cluster_1 + cluster_2 + cluster_3 + cluster_5 + fatalities_log_lag1+NY.GDP.PCAP.CD_log+SP.POP.TOTL_log+v2x_libdem+v2x_clphy+v2x_corr+v2x_rule+v2x_civlib+v2x_neopat, data = df_s)
summary(lm2)

lm3 <- lm(fatalities_log ~ n_protest_events_norm + n_protest_events_norm_lag_1 + n_protest_events_norm_lag_2 + n_protest_events_norm_lag_3 + fatalities_log_lag1 +NY.GDP.PCAP.CD_log+SP.POP.TOTL_log+v2x_libdem+v2x_clphy+v2x_corr+v2x_rule+v2x_civlib+v2x_neopat + as.factor(country), data = df_s)
summary(lm3)

lm4 <- lm(fatalities_log ~ cluster_1 + cluster_2 + cluster_3 + cluster_5 + fatalities_log_lag1 +NY.GDP.PCAP.CD_log+SP.POP.TOTL_log+v2x_libdem+v2x_clphy+v2x_corr+v2x_rule+v2x_civlib+v2x_neopat+ as.factor(country), data = df_s)
summary(lm4)

lm5 <- lm(fatalities_log ~ n_protest_events_norm + n_protest_events_norm_lag_1 + n_protest_events_norm_lag_2 + n_protest_events_norm_lag_3 + cluster_1 + cluster_2 + cluster_3 + cluster_5 +fatalities_log_lag1  +NY.GDP.PCAP.CD_log+SP.POP.TOTL_log+v2x_libdem+v2x_clphy+v2x_corr+v2x_rule+v2x_civlib+v2x_neopat + as.factor(country), data = df_s)
summary(lm5)

# Calculate clustered standard errors
clustered_se1 <- vcovCL(lm1, cluster = ~country)
cl_robust1 <- coeftest(lm1, vcov = clustered_se1)
cl_robust1

clustered_se2 <- vcovCL(lm2, cluster = ~country)
cl_robust2 <- coeftest(lm2, vcov = clustered_se2)
cl_robust2

clustered_se3 <- vcovCL(lm3, cluster = ~country)
cl_robust3 <- coeftest(lm3, vcov = clustered_se3)
cl_robust3

clustered_se4 <- vcovCL(lm4, cluster = ~country)
cl_robust4 <- coeftest(lm4, vcov = clustered_se4)
cl_robust4

clustered_se5 <- vcovCL(lm5, cluster = ~country)
cl_robust5 <- coeftest(lm5, vcov = clustered_se5)
cl_robust5

# F-test 
f_test_lm1 <- linearHypothesis(lm2, c("cluster_11","cluster_21","cluster_31","cluster_51"), vcov = vcovHC(lm2, type = "HC0", cluster = ~ df_s$country))
f_test_lm4 <- linearHypothesis(lm4, c("cluster_11","cluster_21","cluster_31","cluster_51"), vcov = vcovHC(lm4, type = "HC0", cluster = ~ df_s$country))
f_test_lm5 <- linearHypothesis(lm5, c("cluster_11","cluster_21","cluster_31","cluster_51"), vcov = vcovHC(lm5, type = "HC0", cluster = ~ df_s$country))

add_stars <- function(p_value) {
  if (p_value < 0.001) {
    return("***")
  } else if (p_value < 0.01) {
    return("**")
  } else if (p_value < 0.05) {
    return("*")
  } else if (p_value < 0.1) {
    return("o")
  } else {
    return("")
  }
}

f_test_lm1_star <- paste0(round(f_test_lm1$F[2], 2), add_stars(f_test_lm1$`Pr(>F)`[2]))
f_test_lm4_star <- paste0(round(f_test_lm4$F[2], 2), add_stars(f_test_lm4$`Pr(>F)`[2]))
f_test_lm5_star <- paste0(round(f_test_lm5$F[2], 2), add_stars(f_test_lm5$`Pr(>F)`[2]))

stargazer(cl_robust1, cl_robust2, cl_robust3, cl_robust4, cl_robust5, 
          se = list(cl_robust1[,2], cl_robust2[,2], cl_robust3[,2], cl_robust4[,2],cl_robust5[,2]),
          title = "Regression Results with Clustered Standard Errors",
          type = "latex",
          float = FALSE,
          dep.var.caption = 'Dependent variable: Fatalities (log)',
          omit = "as.factor",
          star.cutoffs = c(0.1,0.05, 0.01,0.001), star.char=c('o','*', '**', '***'),
          no.space=T,
          add.lines = list(c("Country Fixed Effects", "No", "No", "Yes", "Yes", "Yes"),
                           c("Clustered by Country", "Yes", "Yes", "Yes", "Yes", "Yes"),
                           c("F-value (joint significance of clusters)", 
                             "", f_test_lm1_star, "", f_test_lm4_star, f_test_lm5_star),
                           c("R-squared", 
                             round(summary(lm1)$r.squared, 3),
                             round(summary(lm2)$r.squared, 3),
                             round(summary(lm3)$r.squared, 3),
                             round(summary(lm4)$r.squared, 3),
                             round(summary(lm5)$r.squared, 3)),
                           c("Number of Observations", 
                             nobs(lm1), nobs(lm2), nobs(lm3), nobs(lm4),nobs(lm5))),
          notes = "Significance levels: o p<0.1; * p<0.05; ** p<0.01; *** p<0.001",
          notes.align = "l",
          notes.append = FALSE,          
          out = "out/regression_results.tex")

cat("\\documentclass{article}\n",
    "\\usepackage{geometry}\n",
    "\\usepackage{graphicx}\n",
    "\\begin{document}\n",
    "\\begin{table}[htbp]\n",
    "\\centering\n",
    "\\resizebox{\\textwidth}{!}{%\n",
    readLines("out/regression_results.tex"),
    "}\n",
    "\\end{table}\n",
    "\\end{document}\n",
    sep = "\n", file = "out/final_table.tex")

