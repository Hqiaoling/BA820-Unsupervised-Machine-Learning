options(stringsAsFactors = FALSE)

library(readr)
library(tidyverse)
library(factoextra)
library(Rtsne)
library(cluster)
library(StatMatch)
train <- read.csv("../Unsupervised-Machine-Learning/project_test.csv")
glimpse(train)
train<-train %>% select(-X, -issue_d, -train, -education)
head(train)
## PCA with only with non birary variable
numeric_data <- train %>% select(emp_length_int, 
                        annual_inc, 
                        loan_amount, 
                        interest_payment_cat,
                        interest_rate,
                        grade,
                        dti,
                        total_pymnt,
                        total_rec_prncp,
                        recoveries,
                        installment
                        )
glimpse(numeric_data)
skimr::skim(numeric_data)
## create a dataset with only binary variable with 17 columns
binary <- train %>% select(loan_condition_cat,munster:application_type_cat_dummy)
glimpse(binary)

## Princinple Components Analysis
## option 2: Keep only numeric variable to fit the pca
train_pca = prcomp(numeric_data, center = TRUE, scale = TRUE)
names(train_pca)

## summary the pca
summary(train_pca)
get_eigenvalue(train_pca)
## compute standard deviation of each principal component
train_dev<-train_pca$sdev
## compute variance
train_var <- train_dev^2
## proportion of variance explained
train_varex <- train_var/sum(train_var)
## scree plot
plot(train_varex, xlab="Principle Components",
        ylab = "Proportion of Variance Explained",
        type = "b")

## cumulative scree plot
plot(cumsum(train_varex), xlab="Principle Components",
     ylab = "Cumulative Proportion of Variance Explained",
     type = "b")

## Another way to show scree plot
fviz_screeplot(train_pca, addlabels = TRUE)

## Option 3: PCA with all variable without scale
pca_all_without_scale = prcomp(train, center = TRUE, scale = FALSE)
get_eigenvalue(pca_all_without_scale)
## Option 4: PCA with all variables
## with binary variable dataset
pca_all_with_scale = prcomp(train, center = TRUE, scale = TRUE)
get_eigenvalue(pca_all_with_scale)
## The result shows that we need to include at least 13 princinple components which can explain 75% of variance, so this is not the 
## ideal option to choose 
## Let's stick with option 2

###### Based the result, we don't think we should do the pca with binary variables. Because the result does not make sense to us####

############---------------------------------------------------------------------------------------------------------##############
## option 5
## scale numeric only, combine to binary variable and do pca
scale_numeric = scale(numeric_data)

## combine with binary data
option5_data = cbind(scale_numeric, binary)
glimpse(option5_data)
## fit the pca model
pca_option5 = prcomp(option5_data, center = TRUE, scale = FALSE)
summary(pca_option5)
get_eigenvalue(pca_option5)

#### The result shows that first four princinple component only explain 64% of variance which is less than the option 2

######################----------------------------------------------------------------------##########################################
### Stick with the option 2 pca model which can explain 74% of variance and with eigenvalue 1, 
### we decide to use this pca option to combine with binary variable for 
## the further analysis
get_eigenvalue(train_pca)

## extract the first princinple component from option 2
pca_fit = predict(train_pca, newdata = train)
pca_fit = pca_fit[,1:4]
pca_fit = as.data.frame(pca_fit)

## combine the pca and binary to one dataframe
train_new <- cbind(binary, pca_fit)
glimpse(train_new)



### Clustering Analysis
## Use silhouette and wss method to get a sense of the number of clusters
## try option 5 PCA and extract first 9 PCA
## use option5 pca to do kmeans
option5_pca9 = pca_option5$x[,1:9]
k = kmeans(option5_pca9, center = 5, iter.max = 25, nstart = 25)
fviz_cluster(k,train)

## use train_new to do kmeans
k1 = kmeans(train_new, center = 3, iter.max = 25, nstart = 25)
fviz_cluster(k1, train)
## Use min and mix to scale entire data to choose pca
maxs <- apply(train, 2, max) 
mins <- apply(train, 2, min)
train.scaled <- as.data.frame(scale(train, center = mins, scale = maxs - mins))
pca_option6 = prcomp(train.scaled, center = TRUE, scale = FALSE)
get_eigenvalue(pca_option6)
summary(pca_option6)
pca6 = predict(pca_option6, newdata = train)
pca6 = pca6[,1:11]
pca6 = as.data.frame(pca6)

## use pca6 to do kmeans clustering
k2 = kmeans(pca6, center = 3, iter.max= 25, nstart = 25)
fviz_cluster(k2, train)
## use scale data to do kmeans clustering
k3 = kmeans(train.scaled, center = 3, iter.max = 25, nstart = 25)
fviz_cluster(k3, train)

## use Rtsne to choose dimention 
## NOTE:  I didn't scale the data, it does this by default
numeric_tsne = Rtsne(numeric_data, verbose = TRUE,
                 max_iter = 500,
                 check_duplicates = FALSE)

num_tsne_proj = numeric_tsne$Y
head(num_tsne_proj)
num_tsne_proj = as.data.frame(num_tsne_proj)
colnames(num_tsne_proj) = c("dim1", "dim2")
numtsne_dum = cbind(num_tsne_proj, binary)
head(numtsne_dum)
numtsne_dum = as.data.frame(numtsne_dum)
cluster_kmeans=kmeans(numtsne_dum, 3, nstart = 25)
numtsne_dum$cl_kmeans = factor(cluster_kmeans$cluster)

plot_k1=plot_cluster(numtsne_dum, "cl_kmeans", "Accent")
plot_k1


#######################---------------------------#####################
## try the train data
train_tsne = Rtsne(train, verbose = TRUE,
                 max_iter = 500,
                 check_duplicates = FALSE)
tsne_proj=train_tsne$Y
head(tsne_proj)
tsne_df = as.data.frame(tsne_proj)
head(tsne_df)
## plot the tsne result
colnames(tsne_df) = c("dim1", "dim2")
ggplot(tsne_df, aes(x=dim1, y=dim2)) +
  geom_point(size=0.25, color = "pink") +
  guides(colour=guide_legend(override.aes=list(size=6))) +
  xlab("") + ylab("") +
  ggtitle("t-SNE") +
  theme_light(base_size=20) +
  theme(axis.text.x=element_blank(),
        axis.text.y=element_blank()) +
  scale_colour_brewer(palette = "Set2")

## Creating k-means clustering model, and assigning the result to the data used to create the tsne
fit_cluster_kmeans=kmeans(tsne_df, 3, nstart = 25)
tsne_df$cl_kmeans = factor(fit_cluster_kmeans$cluster)
##plot the cluster models onto tsne output
plot_cluster=function(data, var_cluster, palette)
{
  ggplot(data, aes_string(x="dim1", y="dim2", color=var_cluster)) +
    geom_point(size=0.25) +
    guides(colour=guide_legend(override.aes=list(size=6))) +
    xlab("") + ylab("") +
    ggtitle("") +
    theme_light(base_size=20) +
    theme(axis.text.x=element_blank(),
          axis.text.y=element_blank(),
          legend.direction = "horizontal", 
          legend.position = "bottom",
          legend.box = "horizontal") + 
    scale_colour_brewer(palette = palette) 
}


plot_k=plot_cluster(tsne_df, "cl_kmeans", "Accent")
plot_k

###########------------------------------###################
## combine result to the train data
final_data = cbind(tsne_df, train)

k4 = kmeans(final_data, centers = 3, nstart = 25)
fviz_cluster(k4, train)

k5 = kmeans(tsne_df, centers = 2, nstart = 25)
fviz_cluster(k5, train)








