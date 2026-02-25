import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Spam Email Detector", page_icon="📧")

st.title("📧 Spam Email Detection App")
st.write("This app uses TF-IDF + Linear SVM to classify emails as Spam or Not Spam.")

# -----------------------------
# Dataset
# -----------------------------
emails = [
    "Congratulations! You’ve won a free iPhone",
    "Claim your lottery prize now",
    "Exclusive deal just for you",
    "Act fast! Limited-time offer",
    "Click here to secure your reward",
    "Win cash prizes instantly by signing up",
    "Limited-time discount on luxury watches",
    "Get rich quick with this secret method",
    "Hello, how are you today",
    "Please find the attached report",
    "Thank you for your support",
    "The project deadline is next week",
    "Can we reschedule the meeting to tomorrow",
    "Your invoice for last month is attached",
    "Looking forward to our call later today",
    "Don’t forget the team lunch tomorrow",
    "Meeting agenda has been updated",
    "Here are the notes from yesterday’s discussion",
    "Please confirm your attendance for the workshop",
    "Let’s finalize the budget proposal by Friday"
]

labels = [1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0]  # 1=Spam, 0=Not Spam

# -----------------------------
# Vectorization
# -----------------------------
vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words='english',
    ngram_range=(1,2),
    max_df=0.9,
    min_df=1
)

X = vectorizer.fit_transform(emails)

# -----------------------------
# Train/Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, labels, test_size=0.25, random_state=42, stratify=labels
)

# -----------------------------
# Train Model
# -----------------------------
svm_model = LinearSVC(C=1.0)
svm_model.fit(X_train, y_train)

y_pred = svm_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

st.subheader("📊 Model Accuracy")
st.info(f"Improved Model Accuracy: {accuracy:.2f}")

# -----------------------------
# User Input
# -----------------------------
st.subheader("✉️ Test a New Email")

new_email = st.text_area("Enter a new email message:")

if st.button("Predict"):
    if new_email.strip() == "":
        st.warning("Please enter an email message.")
    else:
        new_email_vectorized = vectorizer.transform([new_email])
        prediction = svm_model.predict(new_email_vectorized)

        if prediction[0] == 1:
            st.error("🚨 Result: The email is SPAM.")
        else:
            st.success("✅ Result: The email is NOT Spam.")
