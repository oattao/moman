function plotHistory(valLoss, loss, valAcc, trainAcc) {
    var vL = [];
    var tL = [];
    var vA = [];
    var tA = [];

    for (var i=0; i< valLoss.length; i++) {
        vL.push({x: i, y: valLoss[i]});
        tL.push({x: i, y: loss[i]});
        vA.push({x: i, y: valAcc[i]});
        tA.push({x: i, y: trainAcc[i]});
    }

    var chartLoss = new CanvasJS.Chart("lossHistory", {
        title: {text: "Loss"},
        axisX: {title: "Epoch", interval: 1, minimum:0},
        axisY: {title: "Loss value", minimum: 0, maximum: 10},
        legend: {cursor: "pointter", fontSize: 16},
        data: [{name: "Validation loss", type: "line", dataPoints: vL, showInLegend: true},
               {name: "Training loss", type: "line", dataPoints: tL, showInLegend: true}]
    });

    var chartAcc = new CanvasJS.Chart("accHistory", {
        title: {text: "Accuracy"},
        axisX: {title: "Epoch", interval: 1, minimum: 0},
        axisY: {title: "Accuracy (%)", minimum: 0, maximum: 100},
        legend: {cursor: "pointter", fontSize: 16},
        data: [{name: "Validation accuracy", type: "line", dataPoints: vA, showInLegend: true},
               {name: "Training accuracy", type: "line", dataPoints: tA, showInLegend: true}]
    });
    chartLoss.render();
    chartAcc.render();
}

    //when new data come
    $(document).ready(function(){
        var socket = io.connect('http://' + document.domain + ':' + location.port + '/monitortraining');
        socket.on('newdata', function(msg) {
            var n_data = msg.data;
            var n_loss= n_data.loss;
            var n_vloss = n_data.val_loss;
            var n_acc = n_data.accuracy;
            var n_vacc = n_data.val_accuracy;

            // append new data to 
            var length = chartLoss.options.data[0].dataPoints.length;

            chartLoss.options.data[0].dataPoints.push({x: length, y: n_vloss});
            chartLoss.options.data[1].dataPoints.push({x: length, y: n_loss});

            chartAcc.options.data[0].dataPoints.push({x: length, y: n_vacc});
            chartAcc.options.data[1].dataPoints.push({x: length, y: n_acc});
            chartAcc.render();
            chartLoss.render();
        })
    });

    // When finish training
    $(document).ready(function(){
        var socket = io.connect('http://' + document.domain + ':' + location.port + '/monitortraining');
        socket.on('end', function(msg) {
            var rs = msg.finish;
            console.log(rs);
            if (rs == 'ok') {
                location.reload(true);
            }
        })
    });
}