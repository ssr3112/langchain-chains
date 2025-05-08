from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableParallel

load_dotenv()

# Use Gemini for model1
model1 = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

# model2 = ChatAnthropic(model_name='claude-3-7-sonnet-20250219') # Removed ChatAnthropic
# you can use ChatAnthropic
model2 = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

prompt1 = PromptTemplate(
    template='Generate short and simple notes from the following text \n {text}',
    input_variables=['text']
)

prompt2 = PromptTemplate(
    template='Generate 5 short question answers from the following text \n {text}',
    input_variables=['text']
)

prompt3 = PromptTemplate(
    template='Merge the provided notes and quiz into a single document \n notes -> {notes} and quiz -> {quiz}',
    input_variables=['notes', 'quiz']
)

parser = StrOutputParser()

parallel_chain = RunnableParallel({
    'notes': prompt1 | model1 | parser,
    'quiz': prompt2 | model2 | parser
})

# Use Gemini for the merge_chain as well (consistency, or you could use Claude)
merge_chain = prompt3 | model1 | parser

chain = parallel_chain | merge_chain

text = """
Deep learning is a subset of machine learning that uses artificial neural networks to learn from large amounts of data. These neural networks are composed of layers of interconnected nodes that can identify complex patterns and relationships in the data.

Key concepts in deep learning include:

* Artificial Neural Networks: Inspired by the structure of the human brain, these networks consist of interconnected nodes organized in layers.
* Layers: Neural networks are organized into input, hidden, and output layers. Hidden layers allow the network to learn increasingly complex representations of the data.
* Activation Functions: These functions determine whether a node should be activated or not based on its input. Common activation functions include ReLU, sigmoid, and tanh.
* Backpropagation: An algorithm used to train neural networks by iteratively adjusting the weights of the connections between nodes based on the error between the predicted output and the actual output.
* Convolutional Neural Networks (CNNs): A type of neural network specialized for processing grid-like data, such as images.
* Recurrent Neural Networks (RNNs): A type of neural network designed for processing sequential data, such as text or time series.
* Transformers: A more recent type of neural network architecture that has achieved state-of-the-art results in many natural language processing tasks.

Deep learning has achieved significant breakthroughs in various fields, including:

* Computer Vision: Image recognition, object detection, and image segmentation.
* Natural Language Processing (NLP): Machine translation, text summarization, and sentiment analysis.
* Speech Recognition: Converting spoken language into text.
* Robotics: Enabling robots to perceive their environment and make decisions.
* Healthcare: Medical image analysis, drug discovery, and personalized medicine.

Some popular deep learning frameworks include TensorFlow, PyTorch, and Keras.
"""

result = chain.invoke({'text': text})
print(result)

chain.get_graph().print_ascii()
