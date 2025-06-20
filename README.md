# aiops-4uni

1- AIOps Ticket Classifier – Prototype

Show how to utilize basic AI to assist with real‑world IT issues: *“Can a small script automatically decide what type of support ticket we’re dealing with?”*

The result is a very tiny project. No deep learning, no GPU, no large environment. just a prototype.
A real life enterprise model could be my thesis in the future, I can envision embebing the logic of SLAs, financial impact, severity, etc. as an algorithm within the ticket handling, but without re-training the model and as iteractive. Maybe an advanced agent. We shall see.. some theory needs to be explored to comprehend beyond current phase.

2 - File Purpose
aiops_ticket_classifier.py -: All logic: load data → train → save / load → predict.
sample_tickets.csv: Tiny 10‑row dummy dataset for immediate testing. (Swap with your own!)
README.md: You’re reading it.

3 - How It Works

  Input** → historic support tickets in a simple CSV:  

   | description | category |
   |-------------|----------|
   | “Server is down…” | Infrastructure |
   | “VPN won’t connect” | Networking |
   | … | … |

  Training** – the script turns each description into TF‑IDF vectors (a classic, text representation) and fitting Logistic Regression model.

   Prediction** – for a new ticket, the model outputs the most likely category plus a confidence score.

That’s it. No fine‑tuning transformers or massive datasets -just a working prototype.


4 - Next Steps (future learning)
	•	Collect more labelled tickets to improve accuracy. Likely next step to incorporate a publicly available real life ticketing system db.
	•	Experiment with additional features (priority, product line, etc.).
	•	Compare simple models with a pre‑trained transformer (e.g. DistilBERT) once lerning journey evolves.

5 - Will it run on Google Colab?

Yes. The script is pure Python 3 and depends only on **pandas**, **scikit‑learn**, and **joblib** – all either pre‑installed or one‑line installs on Colab. The **“Quick Start on Google Colab”** section above gives copy‑paste commands. No further adaptation is required.

Instructions: Run Instantly on Google Colab

Colab already has Python & scikit‑learn. Open a new notebook and run:
!wget -q https://raw.githubusercontent.com/protode908/aiops-ticket-demo/main/aiops_ticket_classifier.py
!wget -q https://raw.githubusercontent.com/protode908/aiops-ticket-demo/main/sample_tickets.csv

!pip install -q pandas scikit-learn joblib   # usually optional – present by default

# Train on the sample data
!python aiops_ticket_classifier.py train sample_tickets.csv --out mymodel.joblib

# Try a prediction
!python aiops_ticket_classifier.py predict "VPN connection times out" --model mymodel.joblib

6 - License & Credits

MIT License.
Created as a learning prototype by me :)
Some portions of this project were developed with assistance from AI tools such as ChatGPT (OpenAI) and Gemini (Google).
## Acknowledgments

This project was designed and developed as a personal learning exercise, leveraging my experience in scripting and infrastructure. Special thanks to Emiliano Gallo for his valuable help in shaping the ML components of this prototype — especially for guiding the use of `TfidfVectorizer`, the classifier pipeline structure, and conditional confidence reporting using `predict_proba()`. I also used ChatGPT to explore some best practices in scikit-learn and refine implementation details.
