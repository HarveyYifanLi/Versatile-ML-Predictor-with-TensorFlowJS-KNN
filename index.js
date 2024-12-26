require('@tensorflow/tfjs-node'); // use CPU rather than GPU
const tf = require('@tensorflow/tfjs');
const loadCSV = require('./load-csv');
const { parseCustomizedDateString, parseCustomizedTierString } = require('./custom_feature_parsers');
const { getMeanAndVarianceTensors, trainWithKNN } = require('./computation_helpers');

const datasetFile = 'churned_customers_data.csv';

async function main() {
    let { features, labels, testFeatures, testLabels } = await loadCSV(datasetFile, {
        shuffle: false, // shuffle all the rows of the dataset
        splitTest: undefined || 'default', // number of records to use as test dataset (default to 1/5 of all data), the rest are all training dataset,
        useLlmEmbedding: false, // use LLM vector embedding for each column's value along each row
        dataColumns: ['"Renewal ARR"', '"Monthly Amount"', '"Signup Date"', '"Tier at close"', '"Account Count"'], // i.e. the features
        labelColumns: ['"Age"\r'], // i.e. the labels, NOTE on the \r for the return char at the end of the header column. 
        // Including \r is a MUST, IF said column is the LAST column of the header row, else \r is not needed
        converters: { // create custom parsers to be able to parse ANY customized data columns
            '"Signup Date"' : parseCustomizedDateString,
            '"Tier at close"' : parseCustomizedTierString,
        }
    });
    // Obtain the training features tensor (from the features columns of the training dataset)
    featuresTensor = tf.tensor(features);
    // Obtain the training labels tensor (from the labels columns of the training dataset)
    labelsTensor = tf.tensor(labels);
    // Calculate the mean and variance tensors for the features Tensor which will be used for numerical standardization to normalize features data
    const { mean, variance } = getMeanAndVarianceTensors(featuresTensor, 0);

    // console.log('Features: ', features); 
    // console.log('Labels: ', labels);
    console.log('testFeatures: ', testFeatures); 
    console.log('testLabels: ', testLabels);

    // Train model and Fine tune hyperparameter K to improve prediction accuracy
    const enableFineTune = true;

    if(enableFineTune) {
        const topK = 15; // adjust this upper limit of K to try and fine-tune
        let accuracyBucketArray = [];
        
        for (let idx = 1; idx <= topK; idx++) {
            let testResultBucket = {
                'k': null,
                '0_to_10_percent': 0,
                '10_to_30_percent': 0,
                '30_to_50_percent': 0,
                '50_to_70_percent': 0,
                '70_to_90_percent': 0,
                '90_percent_and_above': 0
            }
            
            testFeatures.forEach((testFeature, i) => {
                const result = trainWithKNN(featuresTensor, labelsTensor, tf.tensor(testFeature), idx, mean, variance);
                const errRate = (testLabels[i][0] - result) / testLabels[i][0];
                const absoluteErrRate = Math.abs(errRate * 100);
            
                if(absoluteErrRate >= 0 && absoluteErrRate < 10) {
                    testResultBucket['0_to_10_percent']++;
                }
                else if(absoluteErrRate < 30) {
                    testResultBucket['10_to_30_percent']++;
                }
                else if(absoluteErrRate < 50) {
                    testResultBucket['30_to_50_percent']++;
                }
                else if(absoluteErrRate < 70) {
                    testResultBucket['50_to_70_percent']++;
                }
                else if(absoluteErrRate < 90) {
                    testResultBucket['70_to_90_percent']++;
                }
                else {
                    testResultBucket['90_percent_and_above']++;
                }
            });
        
            testResultBucket['k'] = idx;
        
            accuracyBucketArray.push(testResultBucket);
        }
        
        console.log('\n-----------------------------------------------------------------');
        console.log('The accuracyBucketArray is: ', accuracyBucketArray);
    }

    // Finally, Make prediction using real production features dataset and the fine-tuned/most accurate K value 
    /* 
    const prod_features = [ ... whatever-real-feature-values-are...]
    prod_features.forEach((prodFeature, i) => {
        const prediction = trainWithKNN(featuresTensor, labelsTensor, tf.tensor(prodFeature), 8, mean, variance);

        console.log('\n*****************************************************************');
        console.log(`The KNN predicted result is: ${prediction}`);
    });
    */
}

main();
