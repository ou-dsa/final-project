library(readr)

card.fraud <- read_csv("~/Documents/Ing.Informatica/Exchange/Intelligent Data Analytics/Final/dataset.csv", 
                       col_types = cols(datetime = col_datetime(format = "%Y-%m-%d %H:%M:%S")))

# Time windows
name <- c("1d", "2d", "7d", "30d")
secs <- c(60 * 60 * 24,       # 1 day
          60 * 60 * 24 * 2,   # 2 days
          60 * 60 * 24 * 7,   # 7 days
          60 * 60 * 24 * 30)  # 30 days
time.windows <- data.frame(name, secs)

clients <- unique(card.fraud$tokenized_pan)

for (i in 1:nrow(time.windows)) {
  tw <- time.windows[i, ]
  card.fraud$Id <- seq.int(nrow(card.fraud))
  
  cnt.attr.name <- paste("cnt_", tw$name, sep = "")
  sum.attr.name <- paste("sum_", tw$name, sep = "")

  card.fraud[, cnt.attr.name] <- NA
  card.fraud[, sum.attr.name] <- NA
  
  for (client in clients) {
    client.trxs <- card.fraud[card.fraud$tokenized_pan == client, ]
    
    for (j in 1:nrow(client.trxs)) {
      trx <- client.trxs[j, ]
      current.datetime <- trx$datetime
      related.trxs <- client.trxs[client.trxs$datetime > current.datetime - tw$secs & 
                                    client.trxs$datetime <= current.datetime  &
                                    client.trxs$Id != trx$Id &
                                    client.trxs$is_fraud == 0, ]
      
      card.fraud[card.fraud$Id == trx$Id, cnt.attr.name] <- nrow(related.trxs)
      card.fraud[card.fraud$Id == trx$Id, sum.attr.name] <- sum(related.trxs$amount)
    }
  }
  
  card.fraud <- card.fraud[, !names(card.fraud) %in% c("Id")]
}

issuers <- unique(card.fraud$id_issuer)
card.fraud[, "frd_by_id_issuer"] <- NA
for (issuer in issuers) {
  issuer.frauds <- nrow(card.fraud[card.fraud$is_fraud == 1 & card.fraud$id_issuer == issuer, ])
  issuer.count <- nrow(card.fraud[card.fraud$id_issuer == issuer, ])
  card.fraud$frd_by_id_issuer[card.fraud$id_issuer == issuer] <- issuer.frauds/issuer.count
}

merchants <- unique(card.fraud$id_merchant)
card.fraud[, "frd_by_id_merchant"] <- NA
for (merchant in merchants) {
  merchant.frauds <- nrow(card.fraud[card.fraud$is_fraud == 1 & card.fraud$id_merchant == merchant, ])
  merchant.count <- nrow(card.fraud[card.fraud$id_merchant == merchant, ])
  card.fraud$frd_by_id_merchant[card.fraud$id_merchant == merchant] <- merchant.frauds/merchant.count
}

write.csv(card.fraud,
          file = "~/Documents/Ing.Informatica/Exchange/Intelligent Data Analytics/Final/dataset_agg.csv",
          row.names=FALSE)

ComputeSavings <- function(amounts, pred.values, true.values) {
  predictions <- data.frame(amounts, pred.values, true.values)
  
  costs <- 0
  for (i in 1:nrow(predictions)) {
    pred.value <- predictions$pred.values[i, ]
    true.value <- predictions$true.values[i, ]
    
    if (pred.value == 1) {
      costs <- costs + 20
    } else if (pred.values == 0 & true.value == 1) {
      costs <- costs + predictions$amount[i, ]
    }
  }
  
  savings <- sum(predictions$amounts[predictions$true.values == 1, ]) - costs
  
  return(savings)
}
