const fs = require("fs");
const unidecode = require('unidecode');
const tf = require("@tensorflow/tfjs-node");
const { BPETokenizer } = require("tokenizers");
const readlineSync = require('readline-sync');
let { Tokenizer } = require("tokenizers/bindings/tokenizer");
//const arrayShuffle = require("array-shuffle");
/*const Papa = require("papaparse");
const { data } = Papa.parse(fs.readFileSync("training.csv").toString());
fs.writeFileSync("train.json", JSON.stringify(arrayShuffle(data).slice(0, 100000).map(x => [x[0], x[5]])));*/
function filterString(str) {
    const asciiSafe = unidecode(str).toLowerCase();
    return asciiSafe /*.replace(/[^\w\s\.]/g, "")*/ ;
}
let data = JSON.parse(fs.readFileSync("train.json")).map(x => [(+x[0] - 2) / 2, filterString(x[1].trim())]);
//fs.writeFileSync("vocab.txt", data.map(x => x[1]).join("\n"))
async function main() {
    //const tokenizer = await BPETokenizer.fromOptions();
    //tokenizer.train(["./vocab.txt"], { vocabSize: 4092 });
    //tokenizer.save("./tokenizer.json");
    //const { model } = JSON.parse(fs.readFileSync("tokenizer.json"));
    //fs.writeFileSync("tokenizer/vocab.json", JSON.stringify(model.vocab, undefined, 4));
    //fs.writeFileSync("tokenizer/merges.txt", model.merges.join("\n"));
    const tokenizer = await BPETokenizer.fromOptions({
        vocabFile: "./tokenizer/vocab.json",
        mergesFile: "./tokenizer/merges.txt"
    });
    //data = await Promise.all(data.map(async x => [await tokenizer.encode(x[1]).tokens, x[0]]));
    /*for (let i = 0; i < data.length; i++) {
        console.log(await tokenizer.encode(data[i][1]))
            //data[i][1] = await tokenizer.encode(data[i][1]).tokens;
    }
    console.log(data)*/
    for (let i = 0; i < data.length; i++) {
        const wpEncoded = await tokenizer.encode(data[i][1]);
        data[i][1] = wpEncoded.ids;
    }
    data = data.map(x => [x[1], x[0]]);
    const MESSAGE_SIZE = 25;
    let padded = 0;
    let trimmed = 0;
    for (let i = 0; i < data.length; i++) {
        if (data[i][0].length < 25) {
            padded++;
            while (data[i][0].length < 25) {
                data[i][0].push(4092);
            }
        } else if (data[i][0].length > 25) {
            trimmed++;
            while (data[i][0].length > 25) {
                data[i][0].pop();
            }
        }
        if (data[i][1] === 1) {
            data[i][1] = [1, 0];
        } else {
            data[i][1] = [0, 1];
        }
    }
    const dataTensor = tf.tensor(data.map(x => x[0]));
    const labelTensor = tf.tensor(data.map(x => x[1]));
    //const valDataTensor = tf.tensor(data.map(x => x[0]).slice(90000));
    //const valLabelTensor = tf.tensor(data.map(x => x[1]).slice(90000));
    const model =
        /*tf.sequential({
               layers: [
                   tf.layers.embedding({ inputDim: 4093, outputDim: 128 }),
                   tf.layers.lstm({ units: 512, returnSequences: true }),
                   tf.layers.lstm({ units: 512, returnSequences: true }),
                   tf.layers.lstm({ units: 512, returnSequences: false }),
                   tf.layers.dense({ units: 128, activation: "relu" }),
                   tf.layers.dense({ units: 2, activation: "softmax" })
               ]
           });*/
        await tf.loadLayersModel("file://./model/model.json");
    const TRAIN_MODEL = false;
    async function sentimentAnalysis(str) {
        const tokens = (await tokenizer.encode(str.toLowerCase())).ids;
        while (tokens.length < 25) {
            tokens.push(4092)
        }
        const prediction = model.predict(tf.tensor([tokens])).dataSync();
        return prediction[0] - prediction[1];
    }
    if (TRAIN_MODEL) {
        model.compile({
            optimizer: tf.train.adam(3e-4),
            loss: "categoricalCrossentropy",
            metrics: ["accuracy"],
        });
        model.fit(dataTensor, labelTensor, {
            epochs: 6,
            batchSize: 128,
            callbacks: {
                async onTrainEnd() {
                    await model.save(`file://./model-2`);
                },
                onEpochEnd: tf.callbacks.earlyStopping({ monitor: 'val_acc', patience: 2 })
            }
            /*[tf.callbacks.earlyStopping({ monitor: 'val_acc', patience: 1 }), new tf.CustomCallback({
                async onTrainEnd() {
                    await model.save(`file://./model-2`);
                }
            })]*/
        });
    } else {
        while (true) {
            const input = readlineSync.question("Enter text to analyze: ");
            if (input === "/break") {
                break;
            }
            console.log("\"" + input + "\"" + " was analyzed as " + await sentimentAnalysis(input) + ".")
        }
        //console.log(await sentimentAnalysis("this was not good, not great, and not fun"))
        //console.log(await sentimentAnalysis("this was good, great, and fun"))
        //console.log(await sentimentAnalysis("i hate you"))
    }

}
main();