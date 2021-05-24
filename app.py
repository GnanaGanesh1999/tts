# importing the required libraries
from flask import Flask, render_template, request, redirect, url_for, send_file
from joblib import load
import pandas as pd
import tts
import supervised
from trainer import train

# load the pipeline object
pipeline = load("tamil_sentence_classification.joblib")


# function to get results for a particular text query
def requestResults(tamil_text):
    result = pipeline.predict([tamil_text])
    sentence = "Undefined"
    if result == 0:
        sentence = "declarative"
    elif result == 1:
        sentence = "exclamatory"
    elif result == 2:
        sentence = "imperative"
    elif result == 3:
        sentence = "interrogative"
    return {"type": sentence, "code": result}


# start flask
app = Flask(__name__)


# render default webpage
@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')


# when the post method detect, then redirect to success function
@app.route('/', methods=['POST'])
def get_data():
    if request.method == 'POST':
        text = request.form['text']
        # print(text)
        # print(request.form['type'])
        if request.form['type'] == "identify":
            return redirect(url_for('success', name=text, cont=False))
        if request.form['type'] == "tts":
            with open("data/input.txt", "w", encoding="utf-8") as input_file:
                input_file.write(text)
            path_to_file = './static/output/output.mp3'
            tts.tts()
            return send_file(
                path_to_file,
                mimetype="audio/mp3",
                as_attachment=True, )
        if request.form['type'] == "test":
            return redirect(url_for('train'))

        if request.form['type'] == "postag":
            with open('data/tamil_test.txt', 'w', encoding='utf-8') as f:
                f.write(text)
            return redirect(url_for('pos_tagger', text=text))
        if request.form['type'] == "update":
            train()

    return render_template('index.html')


# get the data for the requested query
@app.route('/success/<name>/<cont>', methods=['POST', 'GET'])
def success(name, cont):
    name = name.strip("TrueFalse")
    return render_template(f"{requestResults(name)['type']}.html", text=name, cont=cont)


@app.route('/save/<text>/<cont>', methods=['POST'])
def save_results(text, cont):
    result = request.form['result']
    result = result.split(" ")
    type_of_sentence = result[1]
    # print(result)
    print(result, text, type_of_sentence, cont)
    text = text.strip("TrueFalse")
    if result[0] == 'correct':
        with open('data/correct.csv', 'a', encoding="utf-8") as file:
            content = text + ',' + type_of_sentence + "\n"
            file.write(content)
        if cont == 'True':
            return redirect(url_for('train'))
        return redirect('/')

    if result[0] == 'wrong':
        with open('data/wrong.csv', 'a', encoding="utf-8") as file:
            content = text + ',' + type_of_sentence + "\n"
            file.write(content)
        return render_template('choose_type.html', text=text, cont=cont)


@app.route('/save-my-result/<text>/<cont>', methods=['POST'])
def save_user_result(text, cont):
    type_of_sentence = request.form['type']
    text = text.strip("TrueFalse")
    with open('data/correct.csv', 'a', encoding="utf-8") as file:
        content = text + ',' + type_of_sentence + "\n"
        file.write(content)

    if cont == 'True':
        return redirect(url_for('train'))
    return redirect('/')


@app.route('/train', methods=['GET'])
def train():
    sentences_data = pd.read_csv('data/data.csv')
    print(len(sentences_data))
    sentence_df = sentences_data.sample()
    print(sentence_df)
    text = sentence_df['sentences'].item()
    sentences_data.drop(sentence_df.index)
    # print(len(sentences_data))
    # sentences_data.to_csv('data/data.csv')
    return redirect(url_for('success', name=text, cont=True))


@app.route('/pos-tags/<text>')
def pos_tagger(text):
    supervised.supervised_tag()
    with open('output/tamil_tags.txt', encoding='utf-8') as f:
        contents = f.read()
        contents = contents.split(" ")
        words = [content.split("_")[0] for content in contents]
        tags = [content.split("_")[1] for content in contents if len(content.split("_")) > 1]
        for tag in tags:
            try:
                tags[tags.index(tag)] += f'  -   {supervised.tag_names[tag]}'
            except KeyError:
                continue
        tags = '<br><br>'.join(tags)
        words = '<br><br>'.join(words)
    return render_template('postags.html', words=words, tags=tags, text=text)


if __name__ == '__main__':
    app.run(debug=True)
