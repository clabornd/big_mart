source("C:/Users/Daniel/Documents/Git Repos/big_mart/PowerBI/requirements.R")

train = read_csv("C:/Users/Daniel/Documents/Git Repos/big_mart/Data/source_data/mart_train.csv")
test = read_csv("C:/Users/Daniel/Documents/Git Repos/big_mart/Data/source_data/mart_test.csv")

train <- train %>% mutate(Item_Fat_Content = ifelse(Item_Fat_Content %in% c("reg", "Regular"), "Regular", "Low"),
                          Outlet_Size = replace(Outlet_Size, is.na(Outlet_Size), "Other"),
                          Item_Category = substr(Item_Identifier, 1,3)) %>%
  group_by(Item_Category) %>%
  mutate(Item_Weight = ifelse(is.na(Item_Weight), mean(Item_Weight, na.rm = TRUE), Item_Weight),
         Item_Visibility = ifelse(Item_Visibility == 0, mean(Item_Visibility), Item_Visibility))

test <- test %>% mutate(Item_Fat_Content = ifelse(Item_Fat_Content %in% c("reg", "Regular"), "Regular", "Low"),
                        Outlet_Size = replace(Outlet_Size, is.na(Outlet_Size), "Other"),
                        Item_Category = substr(Item_Identifier, 1,3)) %>%
  group_by(Item_Category) %>%
  mutate(Item_Weight = ifelse(is.na(Item_Weight), mean(Item_Weight, na.rm = TRUE), Item_Weight),
         Item_Visibility = ifelse(Item_Visibility == 0, mean(Item_Visibility), Item_Visibility))



library(h2o)

h2o.init()

rf_best <- h2o.loadModel("C:\\Users\\Daniel\\Documents\\Git Repos\\big_mart\\Estimators\\rf_grid1_model_764")
gbe_best <- h2o.loadModel("C:\\Users\\Daniel\\Documents\\Git Repos\\big_mart\\Estimators\\gbe_grid_model_1204")

indices = sample(1:nrow(train), floor(.25*nrow(train)))
train_split <- train[-indices,]
valid_split <- train[indices,]

train_final <- as.h2o(test)

preds <- h2o.predict(rf_best, valid_h2o)

y = "Item_Outlet_Sales"

x = names(train_split[-dim(train_split)[2]])

train_h2o = as.h2o(train_split)
valid_h2o = as.h2o(valid_split)

automod <- h2o.automl(x = x, y = y, training_frame = train_h2o, leaderboard_frame = valid_h2o, max_runtime_secs = 30)

automod_leader <- automod@leader

preds <- h2o.predict(object = automod_leader, newdata = train_final)
colnames(preds) <- "Item_Outlet_Sales"