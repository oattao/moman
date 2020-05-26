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