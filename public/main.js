const toAnalyze = document.getElementById("toAnalyze");
const sentimentP = document.getElementById("sentiment");
const analyzeMessage = async() => {
    sentimentP.innerHTML = `Sentiment: Calculating...`;
    $.post("/analyze-message", {
        message: toAnalyze.value
    }, (data) => {
        sentimentP.innerHTML = `Sentiment: ${data.score}`;
    });
};
//toAnalyze.onchange = analyzeMessage;
//toAnalyze.onkeyup = analyzeMessage;
//toAnalyze.paste = analyzeMessage;
//toAnalyze.click = analyzeMessage;
document.getElementById("analyze").onclick = analyzeMessage;