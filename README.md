# Phonepay_analysis
Financial Transaction Classifier &amp; Analytics Dashboard  


An end-to-end Machine Learning and Data Analytics web platform that automatically ingests raw UPI / Bank transaction CSV statements (e.g., PhonePe), cleans metadata headers, classifies transactions into financial categories using a trained ML pipeline, and generates interactive financial analytics.

---

## 🌟 Key Features

- Automated CSV Parsing & Header Detection: Intelligently skips noisy PhonePe metadata headers/disclaimers and extracts clean transaction tables automatically.
- Smart Data Filtering: Cleans out non-essential columns (Transaction ID, UTR, Credit/debit instrument) for a clean presentation.
- Hybrid ML Classification Model: Built with Logistic Regression and TF-IDF Vectorizer trained on 1,600+ real-world transactions (~98.78% accuracy).
- Interactive Web Dashboard: Built using Streamlit to offer instant predictions, interactive category charts, and monthly trend visualization.
- Categorized Data Export: Single-click CSV download feature for processed and categorized transactions.

---

## 📁 Project Structure

`text
├── app.py                      # Main Streamlit Dashboard application
├── save.csv                    # Cleaned dataset used for model training
├── transaction_classifier.pkl  # Trained Logistic Regression model artifact
├── tfidf_vectorizer.pkl        # Fitted TF-IDF Vectorizer artifact
├── README.md                   # Project documentation
└── requirements.txt            # Python dependencies


🚀 Machine Learning Pipeline (Phase Breakdown)
​Phase 1: Data Preprocessing & Cleaning
​Audited 1,637 PhonePe/Bank transactions.
​Handled missing values, normalized text cases, and resolved manual reclassifications. 

​Phase 2: Feature Engineering & Model Training
​Text Vectorization: Applied TfidfVectorizer with ngram_range=(1, 2) to capture single and double-word contextual patterns (e.g., "Sanchi Parlour", "Petro Mart").
​Model: Trained a LogisticRegression classifier with class_weight='balanced' to prevent bias toward dominant categories.
​Accuracy: Achieved 98.78% classification accuracy.
​Artifact Generation: Serialized model into transaction_classifier.pkl and vectorizer into tfidf_vectorizer.pkl.
​
Phase 3: Financial Analytics Engine
​Category-wise Summary: Computes total spending percentage per category.
​Monthly Velocity: Aggregates month-on-month expense trends.
​Payee Insights: Analyzes merchant/payee interaction frequencies.
​
Phase 4: Streamlit Deployment & UI Optimization
​Integrated file ingestion, inference, interactive visualization tabs, and developer profile links in the sidebar.

​👨‍💻 Developer Profile
​Ankit Kumar Ojha
​Email: ankitojha1184@gmail.com
​LinkedIn:https://www.linkedin.com/in/ankit-kumar-ojha-94835b323?utm_source=share_via&utm_content=profile&utm_medium=member_android
