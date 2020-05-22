function changeModel(command) {
    var xhr = new XMLHttpRequest();
    xhr.open("POST", 'http://localhost:8080/changemodel', true);
	xhr.setRequestHeader('Content-Type', 'application/json');
	var model_name = document.getElementById('trainedModelId').innerHTML;
	var parameters = {'command': command, 'model_name': model_name};

	xhr.send(JSON.stringify(parameters));

    // Response from server
	xhr.onload = function() {
		console.log('Response from server:')
		alert('Model is ' + command);
	}
	document.getElementById("resultPanel").style.display = "none";
	document.getElementById("btnTrain").style.display = "block";
}

function showImageTrain() {
	document.getElementById("training").style.display = "block";
	document.getElementById("btnTrain").style.display = "none";
}

function showImage(event) {
	var image = document.getElementById('output');
	image.src = URL.createObjectURL(event.target.files[0]);
	document.getElementById('img_predict').style.display= "none";
	document.getElementById('prediction_result').style.display= "none";
}