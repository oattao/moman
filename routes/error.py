from flask import Blueprint, render_template

error = Blueprint('error', __name__)

@error.errorhandler(404)
def not_found_error(error):
    return render_template('404.html'), 404