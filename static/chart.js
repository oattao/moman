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
			title: {text: "Loss history"},
			axisX: {title: "Epoch", interval: 1, minimum:0},
			axisY: {title: "Loss value", minimum: 0, maximum: 10},
			legend: {cursor: "pointter", fontSize: 16},
			data: [{name: "Validation loss", type: "line", dataPoints: vL, showInLegend: true},
				   {name: "Training loss", type: "line", dataPoints: tL, showInLegend: true}]
		});

		var chartAcc = new CanvasJS.Chart("accHistory", {
			title: {text: "Accuracy history"},
	        axisX: {title: "Epoch", interval: 1, minimum: 0},
			axisY: {title: "Accuracy (%)", minimum: 0, maximum: 100},
			legend: {cursor: "pointter", fontSize: 16},
			data: [{name: "Validation accuracy", type: "line", dataPoints: vA, showInLegend: true},
				   {name: "Training accuracy", type: "line", dataPoints: tA, showInLegend: true}]
		});
		chartLoss.render();
		chartAcc.render();

		document.getElementById("btnTrain").style.display = "none";
}