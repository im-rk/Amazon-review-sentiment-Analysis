import joblib as jb
from flask import Flask, request, render_template
from preprocess_text import preprocess_text  
import pandas as pd
from text_summarization import generate_summary
import matplotlib.pyplot as plt
import base64
from io import BytesIO
import numpy as np
from wordcloud_generator import generate_wordcloud
app = Flask(__name__)

model = jb.load(r"D:\SEM PROJECTS\SEM 2\EOC-2 and MFC-2\Code\sentiment_model.pkl")
vectorizer = jb.load(r"D:\SEM PROJECTS\SEM 2\EOC-2 and MFC-2\Code\tfidf_vectorizer.pkl")

@app.route("/")
def main():
    return render_template("main.html")

@app.route("/single", methods=["POST", "GET"]) 
def index1():
    if request.method == "POST":
        if "text" in request.form and request.form["text"]:
            user_input = request.form["text"]
            processed_input = preprocess_text(user_input)
            transformed_input = vectorizer.transform([processed_input])
            prediction = model.predict(transformed_input)[0]
            #sentiment = "Positive 😀" if prediction == 1 else "Negative 😞"

            return render_template("single.html", user_input=user_input, sentiment=prediction)

    return render_template("single.html")  

@app.route("/multiple", methods=["POST", "GET"]) 
def index2():
    if request.method == "POST":
        if "file" in request.files:
            file = request.files["file"]
            if file.filename.endswith(".csv"):

                file.seek(0)
                df = pd.read_csv(file,encoding="utf-8", delimiter=",")
                if "ReviewContent" in df.columns:
                    df = df.dropna(subset=["ReviewContent"])  
                    df["ReviewContent"] = df["ReviewContent"].astype(str).str.strip()  
                    df = df[df["ReviewContent"] != ""]  
                    df["processed"] = df["ReviewContent"].apply(preprocess_text)

                    if df["processed"].str.strip().eq("").all():  
                        return "No valid reviews to analyze."

                    trans_reviews = vectorizer.transform(df["processed"])
                    df["sentiment"] = model.predict(trans_reviews)


                    # Convert sentiment from 1/0 to "Positive"/"Negative"
                    #df["sentiment"] = df["sentiment"].map({1: "Positive", 0: "Negative"})

                    positive_count = (df["sentiment"] == "Positive").sum()
                    negative_count=(df['sentiment']=="Negative").sum()
                    fig, ax = plt.subplots()
                    ax.bar(["Positive", "Negative"], [positive_count, negative_count], color=["green", "red"])
                    ax.set_ylabel("Number of Reviews")
                    ax.set_title("Sentiment Analysis")

                    # Convert plot to image format
                    img = BytesIO()
                    plt.savefig(img, format="png")
                    img.seek(0)
                    chart_url = base64.b64encode(img.getvalue()).decode()

                    total_reviews = len(df)
                    buy_decision = "BUY ✅" if positive_count / total_reviews > 0.6 else "NOT BUY ❌"
                    summary=generate_summary(file)

                    word_cloud=generate_wordcloud(df["ReviewContent"].tolist())
                    data_frame=df[["ReviewContent","sentiment"]].head(30)
                    data_frame.index=range(1,len(data_frame)+1)
                    return render_template("multiple.html", 
                                           sentiment=buy_decision,
                                             csv_processed=True, 
                                             reviews=data_frame.to_html(),
                                             summary=summary,
                                             positive_count=int(positive_count),
                                             negative_count=int(negative_count),
                                             chart_url=chart_url,
                                             total_reviews=total_reviews,
                                             word_cloud=word_cloud)
                else:
                    return "CSV file should contain a 'reviews' column."
            else:
                return "Only CSV files are allowed."

    return render_template("multiple.html")  

if __name__ == "__main__":
    app.run(debug=True)
