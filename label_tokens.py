from flask import Flask, render_template, request, url_for, redirect
from neo4j.v1 import GraphDatabase, basic_auth
import re
from model.data_utils import CoNLLDataset
from model.ner_model import NERModel
from model.config import Config

driver = GraphDatabase.driver("bolt://localhost", auth=basic_auth("neo4j", "neo"))

app = Flask(__name__)

ingredients_to_label_query = """\
MATCH (ingredient:Ingredient)
WHERE not(exists(ingredient.labelling_done))
RETURN ingredient, apoc.coll.sortNodes([(ingredient)-[:HAS_TOKEN]->(token) | token], "index") AS tokens
ORDER BY rand()
LIMIT 10
"""

config = Config()
model = NERModel(config)
model.build()
model.restore_session(config.dir_model)


def align_data(data):
    spacings = [max([len(seq[i]) for seq in data.values()])
                for i in range(len(data[list(data.keys())[0]]))]
    data_aligned = dict()

    # for each entry, create aligned string
    for key, seq in data.items():
        str_aligned = ""
        for token, spacing in zip(seq, spacings):
            str_aligned += token + " " * (spacing - len(token) + 1)

        data_aligned[key] = str_aligned

    return data_aligned


@app.route("/")
def home():
    with driver.session() as session:
        result = session.run(ingredients_to_label_query)
        ingredients = [{"ingredientId": row["ingredient"]["id"],
                        "value": row["ingredient"]["value"],
                        "tokens": [token for token in row["tokens"]]}
                       for row in result]

    for ingredient in ingredients:
        words_raw = [token["value"] for token in ingredient["tokens"]]
        preds = model.predict(words_raw)
        ingredient["preds"] = preds

        to_print = align_data({"input": words_raw, "output": preds})

        for key, seq in to_print.items():
            model.logger.info(seq)

    return render_template("index.html", ingredients=ingredients)


def extract_label_keys(form):
    keys = [field for field in form if field.startswith("label_")]
    sorted_keys = sorted([(re.match("label_(\d{1,2})", label).groups()[0], label) for label in keys],
                         key=lambda tup: tup[0])
    return [key for index, key in sorted_keys]


label_tokens_query = """\
MATCH (ingredient:Ingredient {id: {id}})
SET ingredient.labelling_done = true
WITH ingredient
MATCH (ingredient)-[:HAS_TOKEN]->(token)
SET token.label = {labels}[token.index]
"""


@app.route("/ingredient/<ingredient_id>", methods=["POST"])
def post_ingredient(ingredient_id):
    label_keys = extract_label_keys(request.form)

    labels = [request.form[key] for key in label_keys]
    print(ingredient_id, labels)

    with driver.session() as session:
        session.run(label_tokens_query, {"labels": labels, "id": ingredient_id})

    return redirect(url_for('home'))
