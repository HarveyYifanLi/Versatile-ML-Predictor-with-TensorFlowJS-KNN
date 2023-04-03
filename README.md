# Versatile-ML-Predictor-with-TensorFlowJS-KNN
This is a highly versatile and portable Supervised Machine Learning Program that utilises `TensorFlow.js` library to implement the `KNN` algorithm to make prediction (i.e. a Label) based on ANY number/dimention of input data Features.
For instance, it can take a csv file with churned customers data and parse ANY number of specified data columns as `Features` 
and make a prediction on another targeted data column, namely a `Label`, such as 'Age' which specifies how long a customer has stayed with
the company before churning.

### Instructions for execution
1. Clone the Repo and cd into that cloned directory
2. Install TensorFlow.js and all other related Dependencies specified in `package.json` via command:
`npm install`
3. Execute the entry point script `index.js` using Node.js runtime via command:
`node index.js` (or if you have installed `nodemon` globally, even better, just run command `nodemon`)

### Further explanations on Prerequisites and customization
1. The program takes a csv file as input. 
A sample file named 'churned_customers_data.csv' with only one row of fake data has been included as a super simple example. 

2. Based on your use cases, however wild that maybe, you simply need to modify the two spots of the program:
<br/>
2a) Based on the headers of your csv file, specify what columns to use as Features and what target column as a Label for which to make a prediction,

i.e. Modify the following two lines in `index.js`:

 `dataColumns: ['"Renewal ARR"', '"Monthly Amount"', '"Signup Date"', '"Tier at close"', '"Account Count"'], // i.e. the features`
 
 `labelColumns: ['"Age"'], // i.e. the labels,`
 
2b) Due to the possibilities that some of data columns in your csv file might contain strings of very specific format (i.e. with arbitrary characters),
all you need to do is to write customized string parsers/helper-functions (and add them to custom_feature_parsers.js) and use them accordingly as part of the `converters`:

i.e. Modify the following in `index.js`:

`converters: { // create custom parsers to be able to parse ANY customized data columns`

`        '"Signup Date"' : parseCustomizedDateString,`
        
`        '"Tier at close"' : parseCustomizedTierString,`
        
`} `
