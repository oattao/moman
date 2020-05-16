from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

@app.route('/')
def manager():
    return render_template('manager.html')

@app.route('/training', methods=['POST'])
def listen():
    parameters = request.json
    print(parameters)
    learning_rate = parameters['lr']
    image_folder = parameters['image_folder']
    model_name = parameters['model_name']
    print('_'*60)
    print(learning_rate)
    print(image_folder)
    print(model_name)
    return jsonify({'status': 0})


if __name__ == "__main__":
    app.run(debug=True, port=8080)

