# tesla ts to MA

tesla<-read.csv("TSLA_2years.csv",header=TRUE)
tesla.ts<-ts(tesla[,2])

plot(tesla.ts,ylab='price',main='Tesla Price Data')
z<-filter(tesla.ts,rep(1/31,31),sides=2)
lines(tesla.MA,col='red')


par(mfrow=c(3,1))
y<-tesla.ts/tesla.MA
plot(y,ylab='scaled price',main='Transformed Tesla Price Data')
acf(na.omit(y),main='Autocorrelation Function of Transformed Tesla Data')
acf(na.omit(y), type='partial',main='Partial ACF of Transformed Tesla Data')

# z<-tesla.MA
par(mfrow=c(3,1))
plot(z,ylab='tesla ma price',main='tesla MA data')
plot(tesla.ts,ylab='price',main='Tesla Price Data')
acf(na.omit(z),main='ACF of Tesla MA')
acf(na.omit(z),type='partial',main='PACF of Tesla MA')
