const MAX_MESSAGE_LENGTH = 240;
data = data.map(x => {
    x[1] = x[1].split("").flatMap(x => {
        //console.log(unidecode(x).split(""));
        return unidecode(x);
    });
    while (x[1].length !== MAX_MESSAGE_LENGTH) {
        x[1].push("empty")
    }
    const old = x[1];
    x[1] = [];
    for (let i = 0; i < old.length; i++) {
        if (old[i] !== "empty") {
            for (let j = 0; j < old[i].length; j++) {
                x[1].push(old[i][j]);
            }
        } else {
            x[1].push(old[i]);
        }
    }
    return [x[0], x[1]];
});
const chars = Array.from(new Set(data.flatMap(x => x[1]))).sort();
const encoding = Object.fromEntries(chars.map((x, i) => [x, i]));
const decoding = Object.fromEntries(chars.map((x, i) => [i, x]));

function oneHotEncode(char) {
    const vec = Array(chars.length).fill(0);
    vec[encoding[char]] = 1;
    return vec;
}
const trainVector = data.map(x => [x[1].map(y => encoding[y]), x[0]]).slice(0, 10000);
const trainData = trainVector.map(x => {
    if (x[0].length === 240) {
        return x[0];
    } else if (x[0].length < 240) {
        while (x[0].length < 240) {
            x[0].push(encoding["empty"]);
        }
        return x[0];
    } else if (x[0].length > 240) {
        while (x[0].length > 240) {
            x[0].pop();
        }
        return x[0];
    }
});
const trainLabels = trainVector.map(x => x[1] === 1 ? [1, 0] : [0, 1]);
const dataTensor = tf.tensor(trainData);
const labelTensor = tf.tensor(trainLabels);
/*const model = tf.sequential({
    layers: [
        tf.layers.embedding({ inputDim: chars.length, outputDim: 32 }),
        tf.layers.lstm({ units: 512, returnSequences: true }),
        tf.layers.lstm({ units: 512, returnSequences: true }),
        tf.layers.lstm({ units: 512, returnSequences: false }),
        tf.layers.dense({ units: 128, activation: "relu" }),
        tf.layers.dense({ units: 2, activation: "softmax" })
    ]
});
model.compile({
    optimizer: tf.train.adam(3e-4),
    loss: "categoricalCrossentropy",
    metrics: ["accuracy"],
});
model.fit(dataTensor, labelTensor, {
        epochs: 10,
        batchSize: 128,
        callbacks: {
            async onTrainEnd() {
                await model.save(`file://./model`);
            }
        }
    })*/
async function main() {
    const model = await tf.loadLayersModel("file://./model/model.json");
    console.log(model.predict(tf.tensor(trainData.slice(0, 10))).arraySync());
}
main();
//console.log(dataTensor.shape);
//console.log(labelTensor.shape);