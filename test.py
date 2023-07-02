from transformers import BertTokenizer, BertForSequenceClassification

# Load the BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
print("ccccc")

# Load the pre-trained BERT model for sequence classification
model = BertForSequenceClassification.from_pretrained('bert-base-uncased')

# Define the long text you want to feed to BERT
long_text = """
Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed non turpis eget justo finibus fringilla ac quis leo. Donec nec eros at nibh semper varius. Maecenas et massa id sem tristique congue. Quisque dictum dapibus nunc ac tincidunt. Nunc iaculis justo vel metus lobortis, et viverra mi consequat. Sed non nisi varius, venenatis urna id, laoreet velit. Integer in ex eget massa feugiat bibendum. Fusce cursus erat sit amet dui iaculis, sit amet congue est fringilla. Nullam venenatis maximus lorem, non scelerisque massa lobortis sed. Sed laoreet, sapien in tincidunt convallis, mauris leo pellentesque dui, a varius tortor nisi eu ex.

Phasellus dictum enim ut nulla eleifend, eget elementum ex posuere. Suspendisse potenti. Aenean commodo pharetra finibus. Nam volutpat odio velit, in fringilla ex congue id. Nullam bibendum congue sagittis. Fusce efficitur eleifend elementum. Cras ut velit eget tortor placerat fringilla nec ac urna. Suspendisse tempus tristique quam, sit amet lacinia nisl lobortis eget. Nam semper vehicula mauris, at aliquet justo gravida sed. Nulla vestibulum efficitur ipsum, at fermentum lacus lobortis ac. Proin fermentum justo eu dolor consequat interdum. Nunc feugiat nulla ac ipsum malesuada iaculis. Sed sed orci a mi cursus ullamcorper. Integer vitae risus a libero posuere viverra.

Etiam sed urna eget est rutrum rhoncus. Nullam et turpis tortor. Aenean in dapibus velit. Phasellus euismod lectus mauris, sed malesuada felis finibus vel. Aliquam tincidunt, dui sed congue facilisis, elit sem dapibus purus, a tincidunt nisl nunc et nisl. Vivamus a interdum tellus, non placerat orci. Etiam faucibus, justo id elementum dapibus, est ex malesuada odio, vitae cursus mi arcu in lacus. Nulla facilisi. Sed nec fermentum metus. Etiam non felis eros. Nam bibendum justo non auctor ultricies. Vestibulum mattis lobortis est nec fringilla. Sed auctor, nisi sit amet scelerisque malesuada, nunc ex tincidunt justo, id commodo justo urna et dui. Curabitur id mauris id arcu fermentum convallis. Suspendisse auctor urna eu enim pharetra aliquam. Sed eu ipsum gravida, aliquet tortor vel, iaculis massa.

... (continued long text)
"""

# Split the long text into sentences
sentences = long_text.split('.')

# Process sentence pairs using BERT tokenizer
encoded_inputs = []
for i in range(len(sentences) - 1):
    input_pair = sentences[i].strip() + '.'  # Add period back to the sentence
    input_pair += ' ' + sentences[i+1].strip()
    encoded_input = tokenizer.encode_plus(
        input_pair,
        add_special_tokens=True,
        truncation=True,
        padding='longest',
        max_length=512,  # Maximum length supported by BERT
        return_tensors='pt'  # Return PyTorch tensors
    )
    encoded_inputs.append(encoded_input)
results=[]
# Perform inference using the BERT model
for encoded_input in encoded_inputs:
    outputs = model(**encoded_input)
    # Process the outputs as per your requirements

# Note: This is a basic example for feeding long text to BERT.
# Depending on your specific task, you may need to adjust the encoding and processing steps accordingly.
print(outputs)