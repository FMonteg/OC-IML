from flask import Flask, request, redirect, url_for, flash, jsonify, render_template
import pickle as p
from gensim.utils import tokenize
import string
import re
import numpy as np




app = Flask(__name__)

def join(prefix_files, num_files, dest_file, read_size):
    # Create a new destination file
    output_file = open(dest_file, 'wb')
    # Get a list of the file parts
    parts = [prefix_files+'{}'.format(i+1) for i in range(num_files)]
    # Go through each portion one by one
    for file in parts:
        # Assemble the full path to the file
        path = file
        # Open the part
        input_file = open(path, 'rb')
        while True:
            # Read all bytes of the part
            byte = input_file.read(read_size)
            # Break out of loop if we are at end of file
            if not byte:
                break
            # Write the bytes to the output file
            output_file.write(byte)
        # Close the input file
        input_file.close()
    # Close the output file
    output_file.close()


def formatage_texte(texte):
    texte = re.sub(r'\<a .*\</a\>', ' ', texte)
    texte = re.sub(r'\<pre.*\</pre\>', ' ', texte)
    texte = re.sub(r'\<code.*\</code\>', ' ', texte)
    texte = re.sub(r'\<[^\>]*\>', ' ', texte)
    texte = re.sub(r'\d+', ' ', texte)
    texte = re.sub(rf"[{re.escape(string.whitespace)}]+", ' ', texte)
    return texte

def get_vect(word, model):
    try:
        return model.wv[word]
    except KeyError:
        return np.zeros((model.vector_size,))

def tokenizer(texte):
    return list(tokenize(texte, deacc=True, lower=True))

def text_vectors(texte, model):
    return sum(get_vect(w, model) for w in tokenizer(texte))



def prediction(texte, classif_file, binary_file):
    texte = formatage_texte(texte)
    
    word_embedding = p.load(open(vector_file, 'rb'))
    w2v = text_vectors(texte, word_embedding)
    del word_embedding

    classifier = p.load(open(classif_file, 'rb'))
    y_pred = classifier.predict(w2v.reshape(1,-1))
    del classifier

    binarizer = p.load(open(binary_file, 'rb'))
    tags = binarizer.inverse_transform(y_pred)
    del binarizer

    return list(tags[0])



vector_file = 'vectorizer.pi'
classif_file = 'classifier.pi'
binary_file = 'binarizer.pi'

join(prefix_files='class_', num_files=7, dest_file=classif_file, read_size = 50000000)
join(prefix_files='vect_', num_files=2, dest_file=vector_file, read_size = 50000000)



@app.route("/")
def hello():
    return render_template("accueil.html")



@app.route('/api_post', methods=['POST'])
def makepred():
    data = request.form["texte"]
    tags = prediction(data, classif_file, binary_file)
    return jsonify(tags)



if __name__ == '__main__':

    app.run(debug=True)
    
