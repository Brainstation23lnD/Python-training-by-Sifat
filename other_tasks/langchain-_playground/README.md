## Message:

Messages are the unit of communication in chat models. They are used to represent input and output of a chat model as well as additional context or metadata that maybe associated with a conversation.

Each message has a role such as User, Assistant and content.

# What is inside the message:
- **Role:** The role of a message: e.g., User, Assistant.
- **Content:** The content of the message. e.g., text and multi-modal data.
- **additional metadata** like id, name, **token usage:**, model specific metadata.

### Role: 
1. **System:** Used to tell the behaviour of a model with additional context.
2. **User:** Representation of Input. User can input and command here.
3. **Assistant:** Response from the model which can include text or a request to invoke tools.
4. **Tool:** A message used to pass the results of a tool invocation back to the model after external data or 
processing has been retrieved.


### Content:
The content of a message text or a list of dictionaries representing multimodal data.

### LangChain Messages

LangChain provides a unified message format that can be used across all chat models, allowing users to work with 
different chat models without worrying about the specific details of the message format used by each model provider.

LangChain messages are Python objects that subclass from a BaseMessage.

The five main message types are:

- **SystemMessage:** corresponds to a system role 
- **HumanMessage:** corresponds to a user role 
- **AIMessage:** corresponds to assistant role 
- **AIMessageChunk:** corresponds to an assistant role, used for streaming responses 
- **ToolMessage:** corresponds to a tool role


## Token
Today's LLMs are typically based on transformer architecture that processes a sequence of units known as token.
The token is a fundamental element that models use to break down input and generate output

### What is actually token?
A token is the basic unit that a language model reads, processes and generates. These units can vary based on how 
the model provided defines them, but in general they could represent:
1. A whole word. e.g., apple 
2. A part of a word. e.g., app
3. Or other linguistics components such as punctuations of spaces
Tokenization totally depends on **tokenizer algorithm** which convert input into tokens.

### How tokens work in a language model:

1. **Input Tokenized:** Model focuses on tokens rather than processing characters or entire sentences directly.
For example, If you are prompting -> "Langchain is cool," it could be tokenized into
["Lang", "Chain", " is", " cool", "!"]
2. **Processing:** the transformer architecture behind these modes processes token sequentially to predict the next token in a sentence.
3. **Output Generation:** The model generates one token one by one.


## What is Embedding Model:
Embedding model transforms human language into a format that machines can understand and compare with speed and accuracy.

## Vector Store:

Langchain vector store object contains methods for adding text and Document objects to store and querying them using 
various similarity matrics.