from flask import Flask, render_template, request
from xyz import convert_audio_to_text

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/submit', methods = ['POST'])
def submit():
    if request.method == 'POST':
        url = request.form['url']
        # print(url)
        html_output = convert_audio_to_text(url)

        # if(url == ""):
        #     return render_template('index.html', message = "Please enter the required field")
        return render_template('success.html', data = html_output)
    
if __name__ == '__main__':
    app.debug =True
    app.run()