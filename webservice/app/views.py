from app import app
from flask import send_file,render_template

import subprocess
import os
import sys

@app.route('/score_img')
def plot_score():
    pwd = os.getcwd()
    try:
        source_dir = os.path.realpath(__file__)
        source_dir = os.path.dirname(source_dir)
        top_dir = os.path.join(source_dir,'..','..')
        os.chdir(top_dir)
        ret = subprocess.run([sys.executable, 'plot_rewards.py'],stdout = subprocess.PIPE)
        filename = ret.stdout.decode().strip()
        return  send_file(filename, mimetype='image/png')
    finally:
        os.chdir(pwd)

@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html')