# Flask Packages
from flask import Flask,render_template, request, url_for
from flask_bootstrap import Bootstrap
from flask_uploads import UploadSet, configure_uploads, IMAGES,DATA, ALL

from werkzeug import secure_filename
import os
import datetime
import time
import icr

app = Flask(__name__)
Bootstrap(app)

# Configuration for File Uploads
files = UploadSet('files', ALL)
app.config['UPLOADED_FILES_DEST'] = 'static/uploadsDB'
configure_uploads(app, files)

@app.route('/')
def index():
    return render_template('index.html')

# Route for our Processing and Details Page
@app.route('/dataupload', methods=['GET','POST'])
def dataupload():
    if request.method == 'POST':

        filedata = request.files['data']
        texttype = request.form['texttype']
        filetype = request.form['filetype']
        feedback = request.form['feedback']
        filename = secure_filename(filedata.filename)
        # os.path.join is used so that paths work in every operating system
        filedata.save(os.path.join('sample_segmentation_data', filename))
        fullfile = os.path.join('sample_segmentation_data', filename)

        # For Time
        timestamp = str(datetime.datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d %H:%M:%S"))

        output_text = icr.intel_char_recog(fullfile, texttype, filetype)

        return render_template('details.html', output_text=output_text,
                filename=filename, timestamp=timestamp, feedback=feedback)

if __name__ == '__main__':
	app.run(debug=True)
