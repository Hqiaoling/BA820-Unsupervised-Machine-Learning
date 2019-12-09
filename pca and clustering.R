options(stringsAsFactors = FALSE)

library(readr)
library(tidyverse)
library(factoextra)
library(Rtsne)
library(cluster)
library(StatMatch)
library(FactoMineR)
loan<- read.csv("../Downloads/loan_final313.csv")

loan <- loan %>% filter(purpose == "credit_card"|purpose == "card"| purpose == "medical"|
                          purpose == "house" | purpose == "small_business"| purpose == "vacation")
loan <- loan %>% filter(home_ownership == "RENT"|home_ownership=="OWN"| home_ownership == "MORTGAGE")
loan <- loan %>% filter(region == "munster"| region == "leinster"|region == "cannught"|region == "Ulster"| region == "Northern Ireland")
unique(loan$home_ownership)

loan_clean <- loan %>% select(-1:-4,-7,-8, -10,-13,-15,-17,-19,-20,-21,-23:-24)

loan_lda <- loan %>% select(-1:-4,-6,-8,-12,-14,-16,-18,-20,-21, -23:-24,-30)
## FAMD to select PCA
loan_clean = loan_clean[-c(64349,73131,80297,91070,122179),]
loan_lda = loan_lda[-c(64349,73131,80297,91070,122179),]
res.famd <- FAMD(loan_clean, graph = FALSE,ncp = 40)
eig.val <- res.famd$eig
eig.val

plot(res.famd, choix = "quanti")
plot(res.famd, choix = "quali")
get_eigenvalue(res.famd)
plot(res.famd, choix = "var")

## Extract PCA to do clustering
pca_fit = predict.FAMD(res.famd, newdata = loan_clean)

head(pca_fit)
pcs = pca_fit$coord[,1:12]


eig.val <- res.famd$eig
barplot(eig.val[, 2], 
        names.arg = 1:nrow(eig.val), 
        main = "Variances Explained by Dimensions (%)",
        xlab = "Principal Dimensions",
        ylab = "Percentage of variances",
        col ="steelblue")
# Add connected line segments to the plot
lines(x = 1:nrow(eig.val), eig.val[, 2], 
      type = "b", pch = 19, col = "red")

## clustering
k = kmeans(pcs, center = 3, iter.max = 25, nstart = 25)
fviz_cluster(k, loan_lda)

## use original data
k_original = kmeans(scale(loan_lda), centers = 3, nstart = 25)
fviz_cluster(k_original, loan_lda)

k_wss_pcs = function(k) {
  km = kmeans(as.data.frame(pca_fit), k, nstart=25, iter=25)
  kwss = km$tot.withinss
  return(kwss)
}

x <- 1:15

wss_vals3pcs <- map_dbl(x, k_wss_pcs)

plot(x, wss_vals3pcs, type = "b", main = "K-Evaluation Using WSS")

## put the cluster back to the data
loan_clean$clusters = factor(k$cluster)
head(loan_clean)
write.csv(loan_clean, file = "data.csv")
## sampling pca to do the clust

SAMP = as.data.frame(sample(1:nrow(pca_fit), 20000))

## Tsne 
SAMP_tsne = Rtsne(SAMP, verbose = TRUE,
                  max_iter = 500,
                  check_duplicates = FALSE)
tsne_proj=SAMP_tsne$Y
tsne_df = as.data.frame(tsne_proj)
head(tsne_df)
dim(tsne_df)
## Creating k-means clustering model, and assigning the result to the data used to create the tsne
ksamp=kmeans(tsne_df, 3, nstart = 25)
tsne_df$cl_kmeans = factor(ksamp$cluster)

## setting 3 clusters as output
fit_hclust=hclust(dist(SAMP))
tsne_df$cl_h = factor(cutree(fit_hclust, k=3))

##plot the cluster models onto tsne output
plot_cluster=function(data, var_cluster, palette)
{
  ggplot(data, aes_string(x="V1", y="V2", color=var_cluster)) +
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
plot_h=plot_cluster(tsne_df, "cl_h", "Set1")

## and finally: putting the plots side by side with gridExtra lib...
library(gridExtra)
grid.arrange(plot_k, plot_h,  ncol=2)



colnames(loan_lda)
