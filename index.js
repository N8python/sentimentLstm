const path = require("path");
const http = require("http");
const express = require("express");
const enforce = require('express-sslify');
const herokuOpen = require("heroku-open");
const bodyParser = require("body-parser");
const publicPath = path.join(__dirname, "/public");
const port = process.env.PORT || 3000;
const app = express();
const { BPETokenizer } = require("tokenizers");
const tf = require("@tensorflow/tfjs-node");
let tokenizer;
let model;
async function sentimentAnalysis(str) {
    const tokens = (await tokenizer.encode(str.toLowerCase())).ids;
    while (tokens.length < 25) {
        tokens.push(4092)
    }
    const prediction = model.predict(tf.tensor([tokens])).dataSync();
    return prediction[0] - prediction[1];
}
(async() => {
    tokenizer = await BPETokenizer.fromOptions({
        vocabFile: "./tokenizer/vocab.json",
        mergesFile: "./tokenizer/merges.txt"
    });
    model = await tf.loadLayersModel("file://./model/model.json");
})()
if (herokuOpen()) {
    app.use(enforce.HTTPS({ trustProtoHeader: true }));
}
app.use(bodyParser.urlencoded());
app.use(express.static(publicPath));
app.post("/analyze-message", async(req, res) => {
    if (tokenizer && model) {
        res.send({
            score: await sentimentAnalysis(req.body.message)
        });
    }
})
const server = http.createServer(app);
server.listen(port, () => {
    console.log(`Server is up on port ${port}`);
});