from flask import Blueprint, Flask, render_template, jsonify

monitor = Blueprint('monitor', __name__)

@monitor.route('/monitortraining', methods=['GET'])
def monitortraining():
	return render_template('monitor_page.html')