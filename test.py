from transformers import BertTokenizer, BertForSequenceClassification

# Load the BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
print("ccccc")

# Load the pre-trained BERT model for sequence classification
model = BertForSequenceClassification.from_pretrained('bert-base-uncased')

# Define the long text you want to feed to BERT
long_text = """
n a quaint coastal town, a young artist named Emma yearned to escape the monotony of her life. 
With dreams of fame and success, she poured her heart into her paintings, 
seeking solace and expression in her art. One stormy night, a mysterious stranger 
named Ethan arrived, captivating Emma with his enigmatic charm and dark past. 
Their passionate love affair ignited, but as their bond deepened, secrets began to unravel.
 Ethan's hidden agenda threatened to tear them apart, while Emma's artistic brilliance attracted the attention of a powerful
   art collector. A tragic twist of fate left Emma devastated and questioning her purpose. Determined to reclaim her 
   lost love and artistic vision, she embarked on a perilous journey of self-discovery. With unwavering determination,
     she faced betrayal, danger, and heartache. Along the way, Emma unearthed hidden talents and forged unexpected alliances.
       In a breathtaking climax, she defied the odds and triumphed over adversity, restoring hope and finding her true artistic voice.
         Emerging from the ashes of her dramatic journey, Emma embraced her destiny as a beacon of inspiration, forever leaving a mark on the world through her art.
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
    results.append(outputs)
    # Process the outputs as per your requirements

# Note: This is a basic example for feeding long text to BERT.
# Depending on your specific task, you may need to adjust the encoding and processing steps accordingly.
print(results)