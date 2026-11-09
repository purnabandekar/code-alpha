
import tkinter as tk
from tkinter import scrolledtext
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ==========================================
# FAQ CHATBOT USING NLP
# ==========================================

# Load spaCy English tokenizer
nlp = spacy.blank("en")

# ------------------------------------------
# FAQ DATASET
# ------------------------------------------

faqs = [
    {
        "question": "What are the admission requirements?",
        "keywords": "admission eligibility apply enrollment documents",
        "answer": "Students need to submit their academic documents, identity proof and completed admission application."
    },
    {
        "question": "What is the examination schedule?",
        "keywords": "exam date examination timetable semester paper",
        "answer": "The examination timetable is published by the college examination department."
    },
    {
        "question": "How can I pay my fees?",
        "keywords": "fees payment online money college charges",
        "answer": "You can pay your fees through the college online payment portal or at the accounts department."
    },
    {
        "question": "What are the library timings?",
        "keywords": "library opening closing books reading hours",
        "answer": "The library is generally open from 9 AM to 5 PM on working days. Contact the library for holiday timings."
    },
    {
        "question": "How can I check my results?",
        "keywords": "result marks grades score semester",
        "answer": "You can check your examination results through your university's official result portal."
    },
    {
        "question": "What is the attendance requirement?",
        "keywords": "attendance percentage lectures classes absent",
        "answer": "Students should follow their university's attendance rules. Check your college department for the applicable minimum percentage."
    },
    {
        "question": "How can I get my student ID card?",
        "keywords": "identity card student id lost card",
        "answer": "Contact your college administration office and submit the required student details to obtain an ID card."
    },
    {
        "question": "Are scholarships available?",
        "keywords": "scholarship financial aid fees support government",
        "answer": "Eligible students can apply for government and institutional scholarships through the relevant scholarship portals."
    },
    {
        "question": "How can I contact the college?",
        "keywords": "contact phone email address office",
        "answer": "You can contact the college administration office through its official website or contact details."
    },
    {
        "question": "Where can I find the class timetable?",
        "keywords": "class schedule lecture timetable time table",
        "answer": "The class timetable is generally available through your department or college notice board."
    }
]

# ------------------------------------------
# NLP TEXT PREPROCESSING
# ------------------------------------------

def preprocess(text):

    doc = nlp(text.lower())

    tokens = [
        token.text
        for token in doc
        if token.is_alpha and not token.is_stop
    ]

    return " ".join(tokens)


# ------------------------------------------
# PREPARE FAQ DATA
# ------------------------------------------

faq_texts = []

for faq in faqs:

    combined_text = (
        faq["question"] + " " + faq["keywords"]
    )

    faq_texts.append(preprocess(combined_text))


# TF-IDF vectorization
vectorizer = TfidfVectorizer(
    ngram_range=(1, 2)
)

faq_vectors = vectorizer.fit_transform(faq_texts)


# ------------------------------------------
# FIND BEST FAQ MATCH
# ------------------------------------------

def get_response(user_question):

    question = user_question.lower().strip()

    if question in ["hi", "hello", "hey", "good morning"]:
        return "Hello! Welcome to the FAQ Chatbot. How can I help you?"

    if question in ["bye", "goodbye", "exit"]:
        return "Thank you for chatting! Have a nice day."

    processed_question = preprocess(user_question)

    if not processed_question:
        return "Please enter a meaningful question."

    # Convert user question into TF-IDF vector
    question_vector = vectorizer.transform(
        [processed_question]
    )

    # Calculate cosine similarity
    similarity_scores = cosine_similarity(
        question_vector,
        faq_vectors
    )[0]

    # Find highest similarity
    best_match = similarity_scores.argmax()

    highest_score = similarity_scores[best_match]

    # Similarity threshold
    if highest_score < 0.12:
        return (
            "Sorry, I couldn't understand your question. "
            "Please try asking about admissions, exams, "
            "fees, attendance, library or scholarships."
        )

    return faqs[best_match]["answer"]


# ==========================================
# CHATBOT GUI
# ==========================================

root = tk.Tk()

root.title("FAQ Chatbot - NLP Assistant")
root.geometry("750x600")
root.configure(bg="#101827")
root.resizable(True, True)

# ------------------------------------------
# HEADER
# ------------------------------------------

header = tk.Frame(
    root,
    bg="#17243a",
    height=90
)

header.pack(fill=tk.X)

title = tk.Label(
    header,
    text="🤖 FAQ CHATBOT",
    font=("Arial", 22, "bold"),
    bg="#17243a",
    fg="#38bdf8"
)

title.pack(pady=(15, 5))

subtitle = tk.Label(
    header,
    text="NLP Based Intelligent Question Answering System",
    font=("Arial", 10),
    bg="#17243a",
    fg="white"
)

subtitle.pack()


# ------------------------------------------
# CHAT DISPLAY
# ------------------------------------------

chat_area = scrolledtext.ScrolledText(
    root,
    wrap=tk.WORD,
    font=("Arial", 12),
    bg="#1e293b",
    fg="white",
    insertbackground="white",
    padx=15,
    pady=15,
    borderwidth=0
)

chat_area.pack(
    fill=tk.BOTH,
    expand=True,
    padx=20,
    pady=15
)

chat_area.tag_config(
    "user",
    foreground="#38bdf8",
    font=("Arial", 11, "bold")
)

chat_area.tag_config(
    "bot",
    foreground="#4ade80",
    font=("Arial", 11, "bold")
)

chat_area.tag_config(
    "message",
    foreground="white"
)

chat_area.config(state=tk.DISABLED)


# ------------------------------------------
# DISPLAY MESSAGE
# ------------------------------------------

def display_message(sender, message, tag):

    chat_area.config(state=tk.NORMAL)

    chat_area.insert(
        tk.END,
        sender + "\n",
        tag
    )

    chat_area.insert(
        tk.END,
        message + "\n\n",
        "message"
    )

    chat_area.config(state=tk.DISABLED)

    chat_area.see(tk.END)


# ------------------------------------------
# SEND MESSAGE
# ------------------------------------------

def send_message(event=None):

    user_question = entry.get().strip()

    if not user_question:
        return

    display_message(
        "You:",
        user_question,
        "user"
    )

    entry.delete(0, tk.END)

    response = get_response(user_question)

    display_message(
        "Chatbot:",
        response,
        "bot"
    )


# ------------------------------------------
# INPUT AREA
# ------------------------------------------

input_frame = tk.Frame(
    root,
    bg="#101827"
)

input_frame.pack(
    fill=tk.X,
    padx=20,
    pady=(0, 20)
)

entry = tk.Entry(
    input_frame,
    font=("Arial", 13),
    bg="#263449",
    fg="white",
    insertbackground="white",
    relief=tk.FLAT
)

entry.pack(
    side=tk.LEFT,
    fill=tk.X,
    expand=True,
    ipady=12,
    padx=(0, 10)
)

entry.bind("<Return>", send_message)


send_button = tk.Button(
    input_frame,
    text="SEND ➤",
    font=("Arial", 11, "bold"),
    bg="#0284c7",
    fg="white",
    activebackground="#0369a1",
    activeforeground="white",
    relief=tk.FLAT,
    cursor="hand2",
    command=send_message
)

send_button.pack(
    side=tk.RIGHT,
    ipadx=15,
    ipady=9
)


# ------------------------------------------
# WELCOME MESSAGE
# ------------------------------------------

display_message(
    "Chatbot:",
    "Hello! Welcome to the FAQ Assistant. "
    "Ask me anything about college admissions, "
    "exams, fees, attendance, library or scholarships.",
    "bot"
)

# Start application
root.mainloop()
