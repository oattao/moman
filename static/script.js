function sendData(event) {
    var xhr = new XMLHttpRequest();
    xhr.open("POST", 'http://127.0.0.1:8080', true);
    xhr.setRequestHeader('Content-Type', 'application/json');
    xhr.send(JSON.stringify({value: 'tenet'}));

    xhr.onload = function() {
        console.log('hello');
        console.log(this.responseText);
        var data = JSON.parse(this.responseText);
        console.log(data);
    }
}
