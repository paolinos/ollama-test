import datasets
from langchain.docstore.document import Document
from langchain_community.retrievers import BM25Retriever
from langchain.tools import Tool
from langchain_core.messages import HumanMessage
from assistant import create_assistant_with_tools

def load_documents():
    """
    Load invitees list and prepare documents
    """
    # Load the dataset
    guest_dataset = datasets.load_dataset("agents-course/unit3-invitees", split="train")

    # Convert dataset entries into Document objects
    docs = [
        Document(
            page_content="\n".join([
                f"Name: {guest['name']}",
                f"Relation: {guest['relation']}",
                f"Description: {guest['description']}",
                f"Email: {guest['email']}"
            ]),
            metadata={"name": guest["name"]}
        )
        for guest in guest_dataset
    ]
    return docs


bm25_retriever = BM25Retriever.from_documents(load_documents())

def guest_info_retriever(query: str) -> str:
    """Retrieves detailed information about gala guests based on their name or relation."""

    results = bm25_retriever.invoke(query)
    if results:
        return "\n\n".join([doc.page_content for doc in results[:3]])
    else:
        return "No matching guest information found."

guest_info_tool = Tool(
    name="guest_info_retriever",
    func=guest_info_retriever,
    description="Retrieves detailed information about gala guests based on their name or relation."
)

if __name__ == "__main__":
    alfred = create_assistant_with_tools([guest_info_tool])

    # Questions about people and need to have the tools
    messages = [HumanMessage(content="Tell me about our guest named 'Lady Ada Lovelace'.")]
    response = alfred.invoke({"messages": messages})

    print("🎩 Alfred's Response:")
    print(response['messages'][-1].content)