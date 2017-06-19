# define diretorio de desenvolvimento
setwd('D:\\temp\\elo7\\data')

# carrega libs
library(tm) #nlp
library(e1071) # machine learning
library(caret) # houldout

# carrega dados (arquivo linux)
data_pos <- readLines('rt-polarity.pos', n = 5331, warn = FALSE)
data_neg <- readLines('rt-polarity.neg', n = 5331, warn = FALSE)

# rotula dados
data_pos <- cbind(data_pos, 'pos')
data_neg <- cbind(data_neg, 'neg')

# mescla conjuntos e atribui nomes
data_rt <- as.data.frame(rbind(data_pos, data_neg))
colnames(data_rt) <- c('text', 'sentiment')
head(data_rt)

# transforma atributos
data_rt$text <- as.character(data_rt$text)
data_rt$sentiment <- as.factor(data_rt$sentiment)
table(data_rt$sentiment)
head(data_rt)

# divide em treino/teste
set.seed(1)
dp <- createDataPartition(data_rt$sentiment, p = 2/3, list = FALSE)

data_rt_train <- data_rt[dp, ]
data_rt_test <- data_rt[-dp, ]

# gera base corpus
corpus <- Corpus(VectorSource(c(data_rt_train$text, data_rt_test$text)))
corpus <- tm_map(corpus, content_transformer(tolower))
corpus <- tm_map(corpus, removeNumbers)
corpus <- tm_map(corpus, removePunctuation)
corpus <- tm_map(corpus, removeWords, c("the", "and", stopwords("english")))
corpus <- tm_map(corpus, stripWhitespace)

# gera matriz de termos
dtm <- DocumentTermMatrix(corpus)
dtm # 100% esparsa

# observa frequencia das palavras
freq <- data.frame(sort(colSums(as.matrix(dtm)), decreasing=TRUE))
barplot(table(freq))

# trata esparsidade
dtm <- removeSparseTerms(dtm, 0.995)
dtm

# prepara para aprendizado
data_words <- as.data.frame(as.matrix(dtm))
data_words_train <- head(data_words, nrow(data_rt_train)) 
data_words_test <- tail(data_words, nrow(data_rt_test))
data_words_train <- cbind(data_words_train, sentiment=data_rt_train$sentiment)

# aprende em naive bayes
nb_model <- naiveBayes(sentiment~., data=data_words_train)
nb_pred <- predict(nb_model, data_words_test)
cm <- table(pred=nb_pred, actual=data_rt_test$sentiment)
acc <- sum(diag(cm))/sum(cm)
acc # 63.6%

# aprende em svm
svm_model <- svm(sentiment~., data=data_words_train)
svm_pred <- predict(svm_model, data_words_test)
cm <- table(pred=svm_pred, actual=data_rt_test$sentiment)
acc <- sum(diag(cm))/sum(cm)
acc # 64.5%

# analise da classificacao
classif <- as.data.frame(cbind(text=data_rt_test[ ,1], sentiment=svm_pred))
classif$text <- as.character(classif$text)
classif$sentiment <- as.factor(classif$sentiment)

classif_pos <- classif[classif$sentiment==2, c('text')]
head(classif_pos, 5)
classif_neg <- classif[classif$sentiment=='1' , c('text')]
head(classif_neg, 5)