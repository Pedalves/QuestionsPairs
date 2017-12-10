# based: https://github.com/h2oai/h2o-3/blob/master/h2o-r/demos/rdemo.word2vec.craigslistjobtitles.R
library(data.table)
library(h2o)
h2o.init(nthreads = -1)

df <- fread("resources/quora_duplicate_questions.tsv", select=c("id","question1","question2","is_duplicate"))

df[,":="(question1=gsub("'|\"|'|"|"|\"|\n|,|\\.|.|\\?|\\+|\\-|\\/|\\=|\\(|\\)|'", "", question1),
          question2=gsub("'|\"|'|"|"|\"|\n|,|\\.|.|\\?|\\+|\\-|\\/|\\=|\\(|\\)|'", "", question2))]
df[,":="(question1=gsub("  ", " ", question1),
          question2=gsub("  ", " ", question2))]

questions <- as.data.table(rbind(df[,.(question=question1)], df[,.(question=question2)]))
questions <- unique(questions)

questions.h2o <- as.h2o(questions, col.types=c("String"))

#########################################################################################################

STOP_WORDS = c("ax","i","you","edu","s","t","m","subject","can","lines","re","what",
               "there","all","we","one","the","a","an","of","or","in","for","by","on",
               "but","is","in","a","not","with","as","was","if","they","are","this","and","it","have",
               "from","at","my","be","by","not","that","to","from","com","org","like","likes","so")

tokenize <- function(sentences, stop.words = STOP_WORDS) {
  tokenized <- h2o.tokenize(sentences, "\\\\W+")
  
  # convert to lower case
  tokenized.lower <- h2o.tolower(tokenized)
  # remove short words (less than 2 characters)
  tokenized.lengths <- h2o.nchar(tokenized.lower)
  tokenized.filtered <- tokenized.lower[is.na(tokenized.lengths) || tokenized.lengths >= 2,]
  # remove words that contain numbers
  tokenized.words <- tokenized.lower[h2o.grep("[0-9]", tokenized.lower, invert = TRUE, output.logical = TRUE),]
  
  # remove stop words
  tokenized.words[is.na(tokenized.words) || (! tokenized.words %in% STOP_WORDS),]
}

#########################################################################################################

print("Break questions into sequence of words")
words <- tokenize(questions.h2o$question)


print("Build word2vec model")
vec_size = 60
w2v.model <- h2o.word2vec(words
                          , model_id = "w2v_model"
                          , vec_size = vec_size
                          , min_word_freq = 5
                          , window_size = 5
                          , init_learning_rate = 0.025
                          , sent_sample_rate = 0
                          , epochs = 1)

h2o.saveModel(object = w2v.model, path = 'model')

print(h2o.findSynonyms(w2v.model, "question", count = 5))

#########################################################################################################

question_all.vecs <- h2o.transform(w2v.model, words, aggregate_method = "NONE")

question_all.vecs <- as.data.table(question_all.vecs)
questions_all <- cbind(questions, question_all.vecs)
df <- merge(df, questions_all, by.x="question1", by.y="question", all.x=TRUE, sort=FALSE)
df <- merge(df, questions_all, by.x="question2", by.y="question", all.x=TRUE, sort=FALSE)
colnames(df)[5:ncol(df)] <- c(paste0("q1_vec_C", 1:vec_size), paste0("q2_vec_C", 1:vec_size))

#fwrite(df, "./h2ow2v_vectors.csv")

library(caTools)
set.seed(123)
df$split = sample.split(df, SplitRatio = 0.8)

drops <- c('split', 'question1', 'question2', 'id')

training_set = subset(df, split == TRUE, select = -c(split, question1, question2, id))
test_set = subset(df, split == FALSE, select = -c(split, question1, question2, id))

train.hex <- as.h2o(training_set)
test.hex <- as.h2o(test_set)

model <- h2o.deeplearning(x = 2:vec_size*2 + 1, y = 1, training_frame = train.hex, epochs = 10)

predictions <- h2o.predict(model, test.hex)

#summary(model)

result <- data.matrix(predictions)
result[result > 0] <- 1
result[result < 0] <- 0

correct <- 0
for(i in 1:nrow(test_set)) {
  if(test_set[i]$is_duplicate == result[i])
  {
    correct <- correct + 1
  }
}

correct/nrow(test_set)

