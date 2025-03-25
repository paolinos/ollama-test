"""
Example of Hugginface
https://huggingface.co/learn/agents-course/unit2/langgraph/first_graph

We’ll implement email processing system, where he needs to:
- Read incoming emails
- Classify them as spam or legitimate
- Draft a preliminary response for legitimate emails
- Send information to user when legitimate (printing only)

This example could use a multi-model, but we're going to use the same model for both steps:
- The first step is to identify whether an email is spam or a valid email.
- The second step is to draft a preliminary response.

# Required packages:
pip install langgraph==0.3.19 langchain-ollama==0.3.0 langchain-core==0.3.48

# Run app
python ./apps/spam_checker.py
"""

import time
from typing import TypedDict, List, Dict, Any, Optional
from langgraph.graph import StateGraph, END, START
#from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
import pprint

# Username of the user
USERNAME_PROMP="Bob"
TITLE_PROMP=f"""As Spam checker, you need to analyze emails and determine if they are spam or legitimate, and whether they require the user's attention.
{USERNAME_PROMP} is a Software Engineer who subscribes to newsletters and social networks.
"""

DRAFT_PROMP="As Spam checker, draft a preliminary response to this email." 

LLM_DEBUG=False

OLLAMA_MODEL="llama3.2:1b"
OLLAMA_HOST="http://localhost:11434" # NOTE: default url of ollama

#---------------------------------------------------------------------------------
#                                                           Email State

class EmailState(TypedDict):
    # The email being processed
    email: Dict[str, Any]  # Contains subject, sender, body, etc.
    
    # Analysis and decisions
    is_spam: Optional[bool]

    spam_reason: Optional[str]

    email_category: Optional[str]
    
    # Response generation
    draft_response: Optional[str]
    
    # Processing metadata
    messages: List[Dict[str, Any]]  # Track conversation with LLM for analysis


#---------------------------------------------------------------------------------
#                                                           Functions
"""
ChatOpenAI requiere a APIKEY so we're going to use local ollama
"""
# Initialize our LLM
# model = ChatOpenAI(temperature=0)

# Remember to run on the container with the ollama model already pulled
# Initialize our LLM
model = ChatOllama(
    model= OLLAMA_MODEL,
    temperature= 0,
    base_url= OLLAMA_HOST
)

def read_email(state: EmailState):
    """Spam Checker reads and logs the incoming email"""
    email = state["email"]
    
    # Here we might do some initial preprocessing
    print(f"Spam Checker is processing an email from {email['sender']} with subject: {email['subject']}")
    
    # No state changes needed here
    return {}

def classify_email(state: EmailState):
    """Spam Checker uses an LLM to determine if the email is spam or legitimate"""
    email = state["email"]
    
    # Prepare our prompt for the LLM
    prompt = f"""{TITLE_PROMP}
    
    Email:
    From: {email['sender']}
    Subject: {email['subject']}
    Body: {email['body']}
    
    
    Analyze the body of the email for:
    - Poor grammar and spelling errors.
    - Excessive use of links to unknown or suspicious domains.
    - Keywords commonly used in spam, such as "free," "money," "urgent," etc.
    - Check for phishing attempts, such as requests for sensitive information (passwords, bank details).

    First, determine if this email is spam. If it is spam, explain why.
    If it is legitimate, categorize it (inquiry, complaint, thank you, request, information, etc.).

    I expect the result using the following format>
    type: spam or not spam
    category: inquiry, complaint, thank you, request, information or None if is spam
    reason: explanation of the decition why is spam or not spam
    """
    
    # Call the LLM
    messages = [HumanMessage(content=prompt)]
    response = model.invoke(messages)
    
    # Simple logic to parse the response (in a real app, you'd want more robust parsing)
    response_text = response.content.lower()

    if LLM_DEBUG: print("LLM response:", response_text)

    is_spam = "spam" in response_text and "not spam" not in response_text
    
    # Extract a reason if it's spam
    spam_reason = None
    if is_spam and "reason:" in response_text:
        spam_reason = response_text.split("reason:")[1].strip()
    
    # Determine category if legitimate
    email_category = None
    if not is_spam:
        categories = ["inquiry", "complaint", "thank you", "request", "information"]
        for category in categories:
            if category in response_text:
                email_category = category
                break
    
    # Update messages for tracking
    new_messages = state.get("messages", []) + [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": response.content}
    ]
    
    # Return state updates
    return {
        "is_spam": is_spam,
        "spam_reason": spam_reason,
        "email_category": email_category,
        "messages": new_messages
    }

def handle_spam(state: EmailState):
    """Spam Checker discards spam email with a note"""
    
    if LLM_DEBUG: print("handle_spam:", state)

    print("Spam Checker has marked the email as spam. The email has been moved to the spam folder.")
    
    # We're done processing this email
    return {}

def drafting_response(state: EmailState):
    """Spam Checker drafts a response for legitimate emails"""

    if LLM_DEBUG: print("drafting_response", state)

    email = state["email"]
    category = state["email_category"] or "general"
    
    
    # Prepare our prompt for the LLM
    prompt = f"""{DRAFT_PROMP}
    
    Email:
    From: {email['sender']}
    Subject: {email['subject']}
    Body: {email['body']}
    
    This email has been categorized as: {category}
    
    Draft a brief, professional response that {USERNAME_PROMP} can review and personalize before sending.
    """
    
    # Call the LLM. Here we can use a different model.
    messages = [HumanMessage(content=prompt)]
    response = model.invoke(messages)
    
    # Update messages for tracking
    new_messages = state.get("messages", []) + [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": response.content}
    ]
    
    # Return state updates
    return {
        "draft_response": response.content,
        "messages": new_messages
    }

def notify_user(state: EmailState):
    """Spam Checker notify the user about the email and presents the draft response"""
    email = state["email"]
    
    print("\n" + "="*50)
    print(f"Sir, you've received an email from {email['sender']}.")
    print(f"Subject: {email['subject']}")
    print(f"Category: {state['email_category']}")
    print("\nI've prepared a draft response for your review:")
    print("-"*50)
    print(state["draft_response"])
    print("="*50 + "\n")
    
    # We're done processing this email
    return {}

def route_email(state: EmailState) -> str:
    """Determine the next step based on spam classification"""
    if state["is_spam"]:
        return "spam"
    else:
        return "legitimate"



#---------------------------------------------------------------------------------
#                                                           LangGraph Settings

# Create the graph
email_graph = StateGraph(EmailState)

# Add nodes
email_graph.add_node("read_email", read_email)
email_graph.add_node("classify_email", classify_email)
email_graph.add_node("handle_spam", handle_spam)
email_graph.add_node("drafting_response", drafting_response)
email_graph.add_node("notify_user", notify_user)

# Add edges - defining the flow
email_graph.add_edge(START, "read_email")
email_graph.add_edge("read_email", "classify_email")

# Add conditional branching from classify_email
email_graph.add_conditional_edges(
    "classify_email",
    route_email,
    {
        "spam": "handle_spam",
        "legitimate": "drafting_response"
    }
)

# Add the final edges
email_graph.add_edge("handle_spam", END)
email_graph.add_edge("drafting_response", "notify_user")
email_graph.add_edge("notify_user", END)

# Compile the graph
compiled_graph = email_graph.compile()


#---------------------------------------------------------------------------------
#                                                           Example email
def get_example_email():
    sender="robert@gmail.com"
    subject = "Catching up after all these times"
    body = f"""Hi {USERNAME_PROMP},
    I hope this email finds you well! It feels like ages since we last connected, and I was reminiscing about the good old times when we'd share laughs over a beer.
    How have things been going for you? I'd love to hear about what's new in your life, work, or anything exciting that you've been up to.
    If you're free sometime soon, it'd be amazing to meet up and relive some of those unforgettable moments. Let's plan a time to catch up—beers on me, like old times!
    Looking forward to hearing from you, {USERNAME_PROMP}. Take care and hope to see you soon!
    Best regards, Robert
    Does this sound like the tone you're aiming for? I can tweak it if you'd like!
    """
    return (sender, subject, body)


#---------------------------------------------------------------------------------
#                                                           Run


if __name__ == "__main__":
    print(TITLE_PROMP)
    print('Please enter the sender (Ex: "some@mail.com" or "Jhon Doe") or leave empty to use the example:')
    sender = input()
    if len(sender) == 0:
        (sender, subject, body) = get_example_email()
    else:
        print('Please enter the subject of the email:')
        subject = input()
        if len(subject) == 0:
            raise "Subject is required."
        print('Please enter the subject of the body: (NOT multiline allowed)')
        body = input()
        if len(body) == 0:
            raise "Body is required."
        
    delta_t = time.time()
    result = compiled_graph.invoke({
        "email": {
            "sender": sender,
            "subject": subject,
            "body": body
        },
        "is_spam": None,
        "spam_reason": None,
        "email_category": None,
        "draft_response": None,
        "messages": []
    })

    print("Result of the Email checker:")
    messages = result["messages"] 
    pprint.pp(messages[len(messages)-1]["content"])
    print(f"Processing time: {time.time() - delta_t} seconds")
