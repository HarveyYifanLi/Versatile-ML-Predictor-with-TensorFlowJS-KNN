const _ = require('lodash');

module.exports.parseCustomizedDateString = function (val) {
    if (val) {
        // console.log(val); // e.g. "05/11/2021"
        const dateString = val.replaceAll('"', ''); // e.g. 05/11/2021

        const [month, day, year] = dateString.split('/');

        const date = new Date(+year, month - 1, +day);

        const timestampInSeconds = Math.floor(date.getTime() / 1000); // e.g. 1650080712

        return timestampInSeconds;
    }

    return 0;
}

module.exports.parseCustomizedTierString = function (val) {
    if (val) {
        // console.log(val); // e.g. "Tier 3"
        let [prefix, tierNumber] = val.replaceAll('"', '').split(' ');
        
        tierNumber = parseFloat(tierNumber);

        return _.isNaN(tierNumber) ? 0 : tierNumber;
    }

    return 0;
}
